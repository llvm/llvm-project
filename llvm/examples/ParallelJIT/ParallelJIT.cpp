//===-- examples/ParallelJIT/ParallelJIT.cpp - Exercise threaded-safe JIT -===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Parallel JIT
//
// This test program creates two LLVM functions then calls them from three
// separate threads.  It requires the pthreads library.
// The three threads are created and then block waiting on a condition variable.
// Once all threads are blocked on the conditional variable, the main thread
// wakes them up. This complicated work is performed so that all three threads
// call into the JIT at the same time (or the best possible approximation of the
// same time). This test had assertion errors until I got the locking right.

#include <pthread.h>
#include "llvm/Module.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Instructions.h"
#include "llvm/ModuleProvider.h"
#include "llvm/ExecutionEngine/JIT.h"
#include "llvm/ExecutionEngine/Interpreter.h"
#include "llvm/ExecutionEngine/GenericValue.h"
#include <iostream>
using namespace llvm;

static Function* createAdd1(Module *M) {
  // Create the add1 function entry and insert this entry into module M.  The
  // function will have a return type of "int" and take an argument of "int".
  // The '0' terminates the list of argument types.
  Function *Add1F =
    cast<Function>(M->getOrInsertFunction("add1", Type::Int32Ty, Type::Int32Ty,
                                          (Type *)0));

  // Add a basic block to the function. As before, it automatically inserts
  // because of the last argument.
  BasicBlock *BB = BasicBlock::Create("EntryBlock", Add1F);

  // Get pointers to the constant `1'.
  Value *One = ConstantInt::get(Type::Int32Ty, 1);

  // Get pointers to the integer argument of the add1 function...
  assert(Add1F->arg_begin() != Add1F->arg_end()); // Make sure there's an arg
  Argument *ArgX = Add1F->arg_begin();  // Get the arg
  ArgX->setName("AnArg");            // Give it a nice symbolic name for fun.

  // Create the add instruction, inserting it into the end of BB.
  Instruction *Add = BinaryOperator::CreateAdd(One, ArgX, "addresult", BB);

  // Create the return instruction and add it to the basic block
  ReturnInst::Create(Add, BB);

  // Now, function add1 is ready.
  return Add1F;
}

static Function *CreateFibFunction(Module *M) {
  // Create the fib function and insert it into module M.  This function is said
  // to return an int and take an int parameter.
  Function *FibF = 
    cast<Function>(M->getOrInsertFunction("fib", Type::Int32Ty, Type::Int32Ty,
                                          (Type *)0));

  // Add a basic block to the function.
  BasicBlock *BB = BasicBlock::Create("EntryBlock", FibF);

  // Get pointers to the constants.
  Value *One = ConstantInt::get(Type::Int32Ty, 1);
  Value *Two = ConstantInt::get(Type::Int32Ty, 2);

  // Get pointer to the integer argument of the add1 function...
  Argument *ArgX = FibF->arg_begin();   // Get the arg.
  ArgX->setName("AnArg");            // Give it a nice symbolic name for fun.

  // Create the true_block.
  BasicBlock *RetBB = BasicBlock::Create("return", FibF);
  // Create an exit block.
  BasicBlock* RecurseBB = BasicBlock::Create("recurse", FibF);

  // Create the "if (arg < 2) goto exitbb"
  Value *CondInst = new ICmpInst(ICmpInst::ICMP_SLE, ArgX, Two, "cond", BB);
  BranchInst::Create(RetBB, RecurseBB, CondInst, BB);

  // Create: ret int 1
  ReturnInst::Create(One, RetBB);

  // create fib(x-1)
  Value *Sub = BinaryOperator::CreateSub(ArgX, One, "arg", RecurseBB);
  Value *CallFibX1 = CallInst::Create(FibF, Sub, "fibx1", RecurseBB);

  // create fib(x-2)
  Sub = BinaryOperator::CreateSub(ArgX, Two, "arg", RecurseBB);
  Value *CallFibX2 = CallInst::Create(FibF, Sub, "fibx2", RecurseBB);

  // fib(x-1)+fib(x-2)
  Value *Sum =
    BinaryOperator::CreateAdd(CallFibX1, CallFibX2, "addresult", RecurseBB);

  // Create the return instruction and add it to the basic block
  ReturnInst::Create(Sum, RecurseBB);

  return FibF;
}

struct threadParams {
  ExecutionEngine* EE;
  Function* F;
  int value;
};

// We block the subthreads just before they begin to execute:
// we want all of them to call into the JIT at the same time,
// to verify that the locking is working correctly.
class WaitForThreads
{
public:
  WaitForThreads()
  {
    n = 0;
    waitFor = 0;

    int result = pthread_cond_init( &condition, NULL );
    assert( result == 0 );

    result = pthread_mutex_init( &mutex, NULL );
    assert( result == 0 );
  }

  ~WaitForThreads()
  {
    int result = pthread_cond_destroy( &condition );
    assert( result == 0 );

    result = pthread_mutex_destroy( &mutex );
    assert( result == 0 );
  }

  // All threads will stop here until another thread calls releaseThreads
  void block()
  {
    int result = pthread_mutex_lock( &mutex );
    assert( result == 0 );
    n ++;
    //~ std::cout << "block() n " << n << " waitFor " << waitFor << std::endl;

    assert( waitFor == 0 || n <= waitFor );
    if ( waitFor > 0 && n == waitFor )
    {
      // There are enough threads blocked that we can release all of them
      std::cout << "Unblocking threads from block()" << std::endl;
      unblockThreads();
    }
    else
    {
      // We just need to wait until someone unblocks us
      result = pthread_cond_wait( &condition, &mutex );
      assert( result == 0 );
    }

    // unlock the mutex before returning
    result = pthread_mutex_unlock( &mutex );
    assert( result == 0 );
  }

  // If there are num or more threads blocked, it will signal them all
  // Otherwise, this thread blocks until there are enough OTHER threads
  // blocked
  void releaseThreads( size_t num )
  {
    int result = pthread_mutex_lock( &mutex );
    assert( result == 0 );

    if ( n >= num ) {
      std::cout << "Unblocking threads from releaseThreads()" << std::endl;
      unblockThreads();
    }
    else
    {
      waitFor = num;
      pthread_cond_wait( &condition, &mutex );
    }

    // unlock the mutex before returning
    result = pthread_mutex_unlock( &mutex );
    assert( result == 0 );
  }

private:
  void unblockThreads()
  {
    // Reset the counters to zero: this way, if any new threads
    // enter while threads are exiting, they will block instead
    // of triggering a new release of threads
    n = 0;

    // Reset waitFor to zero: this way, if waitFor threads enter
    // while threads are exiting, they will block instead of
    // triggering a new release of threads
    waitFor = 0;

    int result = pthread_cond_broadcast( &condition );
    assert(result == 0); result=result;
  }

  size_t n;
  size_t waitFor;
  pthread_cond_t condition;
  pthread_mutex_t mutex;
};

static WaitForThreads synchronize;

void* callFunc( void* param )
{
  struct threadParams* p = (struct threadParams*) param;

  // Call the `foo' function with no arguments:
  std::vector<GenericValue> Args(1);
  Args[0].IntVal = APInt(32, p->value);

  synchronize.block(); // wait until other threads are at this point
  GenericValue gv = p->EE->runFunction(p->F, Args);

  return (void*)(intptr_t)gv.IntVal.getZExtValue();
}

int main()
{
  // Create some module to put our function into it.
  Module *M = new Module("test");

  Function* add1F = createAdd1( M );
  Function* fibF = CreateFibFunction( M );

  // Now we create the JIT.
  ExistingModuleProvider* MP = new ExistingModuleProvider(M);
  ExecutionEngine* EE = ExecutionEngine::create(MP, false);

  //~ std::cout << "We just constructed this LLVM module:\n\n" << *M;
  //~ std::cout << "\n\nRunning foo: " << std::flush;

  // Create one thread for add1 and two threads for fib
  struct threadParams add1 = { EE, add1F, 1000 };
  struct threadParams fib1 = { EE, fibF, 39 };
  struct threadParams fib2 = { EE, fibF, 42 };

  pthread_t add1Thread;
  int result = pthread_create( &add1Thread, NULL, callFunc, &add1 );
  if ( result != 0 ) {
          std::cerr << "Could not create thread" << std::endl;
          return 1;
  }

  pthread_t fibThread1;
  result = pthread_create( &fibThread1, NULL, callFunc, &fib1 );
  if ( result != 0 ) {
          std::cerr << "Could not create thread" << std::endl;
          return 1;
  }

  pthread_t fibThread2;
  result = pthread_create( &fibThread2, NULL, callFunc, &fib2 );
  if ( result != 0 ) {
          std::cerr << "Could not create thread" << std::endl;
          return 1;
  }

  synchronize.releaseThreads(3); // wait until other threads are at this point

  void* returnValue;
  result = pthread_join( add1Thread, &returnValue );
  if ( result != 0 ) {
          std::cerr << "Could not join thread" << std::endl;
          return 1;
  }
  std::cout << "Add1 returned " << intptr_t(returnValue) << std::endl;

  result = pthread_join( fibThread1, &returnValue );
  if ( result != 0 ) {
          std::cerr << "Could not join thread" << std::endl;
          return 1;
  }
  std::cout << "Fib1 returned " << intptr_t(returnValue) << std::endl;

  result = pthread_join( fibThread2, &returnValue );
  if ( result != 0 ) {
          std::cerr << "Could not join thread" << std::endl;
          return 1;
  }
  std::cout << "Fib2 returned " << intptr_t(returnValue) << std::endl;

  return 0;
}
