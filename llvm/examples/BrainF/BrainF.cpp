//===-- BrainF.cpp - BrainF compiler example ----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===--------------------------------------------------------------------===//
//
// This class compiles the BrainF language into LLVM assembly.
//
// The BrainF language has 8 commands:
// Command   Equivalent C    Action
// -------   ------------    ------
// ,         *h=getchar();   Read a character from stdin, 255 on EOF
// .         putchar(*h);    Write a character to stdout
// -         --*h;           Decrement tape
// +         ++*h;           Increment tape
// <         --h;            Move head left
// >         ++h;            Move head right
// [         while(*h) {     Start loop
// ]         }               End loop
//
//===--------------------------------------------------------------------===//

#include "BrainF.h"
#include "llvm/Constants.h"
#include "llvm/Intrinsics.h"
#include "llvm/ADT/STLExtras.h"
#include <iostream>
using namespace llvm;

//Set the constants for naming
const char *BrainF::tapereg = "tape";
const char *BrainF::headreg = "head";
const char *BrainF::label   = "brainf";
const char *BrainF::testreg = "test";

Module *BrainF::parse(std::istream *in1, int mem, CompileFlags cf) {
  in       = in1;
  memtotal = mem;
  comflag  = cf;

  header();
  readloop(0, 0, 0);
  delete builder;
  return module;
}

void BrainF::header() {
  module = new Module("BrainF");

  //Function prototypes

  //declare void @llvm.memset.i32(i8 *, i8, i32, i32)
  const Type *Tys[] = { Type::Int32Ty };
  Function *memset_func = Intrinsic::getDeclaration(module, Intrinsic::memset,
                                                    Tys, 1);

  //declare i32 @getchar()
  getchar_func = cast<Function>(module->
    getOrInsertFunction("getchar", IntegerType::Int32Ty, NULL));

  //declare i32 @putchar(i32)
  putchar_func = cast<Function>(module->
    getOrInsertFunction("putchar", IntegerType::Int32Ty,
                        IntegerType::Int32Ty, NULL));


  //Function header

  //define void @brainf()
  brainf_func = cast<Function>(module->
    getOrInsertFunction("brainf", Type::VoidTy, NULL));

  builder = new IRBuilder<>(BasicBlock::Create(label, brainf_func));

  //%arr = malloc i8, i32 %d
  ConstantInt *val_mem = ConstantInt::get(APInt(32, memtotal));
  ptr_arr = builder->CreateMalloc(IntegerType::Int8Ty, val_mem, "arr");

  //call void @llvm.memset.i32(i8 *%arr, i8 0, i32 %d, i32 1)
  {
    Value *memset_params[] = {
      ptr_arr,
      ConstantInt::get(APInt(8, 0)),
      val_mem,
      ConstantInt::get(APInt(32, 1))
    };

    CallInst *memset_call = builder->
      CreateCall(memset_func, memset_params, array_endof(memset_params));
    memset_call->setTailCall(false);
  }

  //%arrmax = getelementptr i8 *%arr, i32 %d
  if (comflag & flag_arraybounds) {
    ptr_arrmax = builder->
      CreateGEP(ptr_arr, ConstantInt::get(APInt(32, memtotal)), "arrmax");
  }

  //%head.%d = getelementptr i8 *%arr, i32 %d
  curhead = builder->CreateGEP(ptr_arr,
                               ConstantInt::get(APInt(32, memtotal/2)),
                               headreg);



  //Function footer

  //brainf.end:
  endbb = BasicBlock::Create(label, brainf_func);

  //free i8 *%arr
  new FreeInst(ptr_arr, endbb);

  //ret void
  ReturnInst::Create(endbb);



  //Error block for array out of bounds
  if (comflag & flag_arraybounds)
  {
    //@aberrormsg = internal constant [%d x i8] c"\00"
    Constant *msg_0 = ConstantArray::
      get("Error: The head has left the tape.", true);

    GlobalVariable *aberrormsg = new GlobalVariable(
      msg_0->getType(),
      true,
      GlobalValue::InternalLinkage,
      msg_0,
      "aberrormsg",
      module);

    //declare i32 @puts(i8 *)
    Function *puts_func = cast<Function>(module->
      getOrInsertFunction("puts", IntegerType::Int32Ty,
                          PointerType::getUnqual(IntegerType::Int8Ty), NULL));

    //brainf.aberror:
    aberrorbb = BasicBlock::Create(label, brainf_func);

    //call i32 @puts(i8 *getelementptr([%d x i8] *@aberrormsg, i32 0, i32 0))
    {
      Constant *zero_32 = Constant::getNullValue(IntegerType::Int32Ty);

      Constant *gep_params[] = {
        zero_32,
        zero_32
      };

      Constant *msgptr = ConstantExpr::
        getGetElementPtr(aberrormsg, gep_params,
                         array_lengthof(gep_params));

      Value *puts_params[] = {
        msgptr
      };

      CallInst *puts_call =
        CallInst::Create(puts_func,
                         puts_params, array_endof(puts_params),
                         "", aberrorbb);
      puts_call->setTailCall(false);
    }

    //br label %brainf.end
    BranchInst::Create(endbb, aberrorbb);
  }
}

void BrainF::readloop(PHINode *phi, BasicBlock *oldbb, BasicBlock *testbb) {
  Symbol cursym = SYM_NONE;
  int curvalue = 0;
  Symbol nextsym = SYM_NONE;
  int nextvalue = 0;
  char c;
  int loop;
  int direction;

  while(cursym != SYM_EOF && cursym != SYM_ENDLOOP) {
    // Write out commands
    switch(cursym) {
      case SYM_NONE:
        // Do nothing
        break;

      case SYM_READ:
        {
          //%tape.%d = call i32 @getchar()
          CallInst *getchar_call = builder->CreateCall(getchar_func, tapereg);
          getchar_call->setTailCall(false);
          Value *tape_0 = getchar_call;

          //%tape.%d = trunc i32 %tape.%d to i8
          Value *tape_1 = builder->
            CreateTrunc(tape_0, IntegerType::Int8Ty, tapereg);

          //store i8 %tape.%d, i8 *%head.%d
          builder->CreateStore(tape_1, curhead);
        }
        break;

      case SYM_WRITE:
        {
          //%tape.%d = load i8 *%head.%d
          LoadInst *tape_0 = builder->CreateLoad(curhead, tapereg);

          //%tape.%d = sext i8 %tape.%d to i32
          Value *tape_1 = builder->
            CreateSExt(tape_0, IntegerType::Int32Ty, tapereg);

          //call i32 @putchar(i32 %tape.%d)
          Value *putchar_params[] = {
            tape_1
          };
          CallInst *putchar_call = builder->
            CreateCall(putchar_func,
                       putchar_params, array_endof(putchar_params));
          putchar_call->setTailCall(false);
        }
        break;

      case SYM_MOVE:
        {
          //%head.%d = getelementptr i8 *%head.%d, i32 %d
          curhead = builder->
            CreateGEP(curhead, ConstantInt::get(APInt(32, curvalue)),
                      headreg);

          //Error block for array out of bounds
          if (comflag & flag_arraybounds)
          {
            //%test.%d = icmp uge i8 *%head.%d, %arrmax
            Value *test_0 = builder->
              CreateICmpUGE(curhead, ptr_arrmax, testreg);

            //%test.%d = icmp ult i8 *%head.%d, %arr
            Value *test_1 = builder->
              CreateICmpULT(curhead, ptr_arr, testreg);

            //%test.%d = or i1 %test.%d, %test.%d
            Value *test_2 = builder->
              CreateOr(test_0, test_1, testreg);

            //br i1 %test.%d, label %main.%d, label %main.%d
            BasicBlock *nextbb = BasicBlock::Create(label, brainf_func);
            builder->CreateCondBr(test_2, aberrorbb, nextbb);

            //main.%d:
            builder->SetInsertPoint(nextbb);
          }
        }
        break;

      case SYM_CHANGE:
        {
          //%tape.%d = load i8 *%head.%d
          LoadInst *tape_0 = builder->CreateLoad(curhead, tapereg);

          //%tape.%d = add i8 %tape.%d, %d
          Value *tape_1 = builder->
            CreateAdd(tape_0, ConstantInt::get(APInt(8, curvalue)), tapereg);

          //store i8 %tape.%d, i8 *%head.%d\n"
          builder->CreateStore(tape_1, curhead);
        }
        break;

      case SYM_LOOP:
        {
          //br label %main.%d
          BasicBlock *testbb = BasicBlock::Create(label, brainf_func);
          builder->CreateBr(testbb);

          //main.%d:
          BasicBlock *bb_0 = builder->GetInsertBlock();
          BasicBlock *bb_1 = BasicBlock::Create(label, brainf_func);
          builder->SetInsertPoint(bb_1);

          // Make part of PHI instruction now, wait until end of loop to finish
          PHINode *phi_0 =
            PHINode::Create(PointerType::getUnqual(IntegerType::Int8Ty),
                            headreg, testbb);
          phi_0->reserveOperandSpace(2);
          phi_0->addIncoming(curhead, bb_0);
          curhead = phi_0;

          readloop(phi_0, bb_1, testbb);
        }
        break;

      default:
        std::cerr << "Error: Unknown symbol.\n";
        abort();
        break;
    }

    cursym = nextsym;
    curvalue = nextvalue;
    nextsym = SYM_NONE;

    // Reading stdin loop
    loop = (cursym == SYM_NONE)
        || (cursym == SYM_MOVE)
        || (cursym == SYM_CHANGE);
    while(loop) {
      *in>>c;
      if (in->eof()) {
        if (cursym == SYM_NONE) {
          cursym = SYM_EOF;
        } else {
          nextsym = SYM_EOF;
        }
        loop = 0;
      } else {
        direction = 1;
        switch(c) {
          case '-':
            direction = -1;
            // Fall through

          case '+':
            if (cursym == SYM_CHANGE) {
              curvalue += direction;
              // loop = 1
            } else {
              if (cursym == SYM_NONE) {
                cursym = SYM_CHANGE;
                curvalue = direction;
                // loop = 1
              } else {
                nextsym = SYM_CHANGE;
                nextvalue = direction;
                loop = 0;
              }
            }
            break;

          case '<':
            direction = -1;
            // Fall through

          case '>':
            if (cursym == SYM_MOVE) {
              curvalue += direction;
              // loop = 1
            } else {
              if (cursym == SYM_NONE) {
                cursym = SYM_MOVE;
                curvalue = direction;
                // loop = 1
              } else {
                nextsym = SYM_MOVE;
                nextvalue = direction;
                loop = 0;
              }
            }
            break;

          case ',':
            if (cursym == SYM_NONE) {
              cursym = SYM_READ;
            } else {
              nextsym = SYM_READ;
            }
            loop = 0;
            break;

          case '.':
            if (cursym == SYM_NONE) {
              cursym = SYM_WRITE;
            } else {
              nextsym = SYM_WRITE;
            }
            loop = 0;
            break;

          case '[':
            if (cursym == SYM_NONE) {
              cursym = SYM_LOOP;
            } else {
              nextsym = SYM_LOOP;
            }
            loop = 0;
            break;

          case ']':
            if (cursym == SYM_NONE) {
              cursym = SYM_ENDLOOP;
            } else {
              nextsym = SYM_ENDLOOP;
            }
            loop = 0;
            break;

          // Ignore other characters
          default:
            break;
        }
      }
    }
  }

  if (cursym == SYM_ENDLOOP) {
    if (!phi) {
      std::cerr << "Error: Extra ']'\n";
      abort();
    }

    // Write loop test
    {
      //br label %main.%d
      builder->CreateBr(testbb);

      //main.%d:

      //%head.%d = phi i8 *[%head.%d, %main.%d], [%head.%d, %main.%d]
      //Finish phi made at beginning of loop
      phi->addIncoming(curhead, builder->GetInsertBlock());
      Value *head_0 = phi;

      //%tape.%d = load i8 *%head.%d
      LoadInst *tape_0 = new LoadInst(head_0, tapereg, testbb);

      //%test.%d = icmp eq i8 %tape.%d, 0
      ICmpInst *test_0 = new ICmpInst(ICmpInst::ICMP_EQ, tape_0,
                                      ConstantInt::get(APInt(8, 0)), testreg,
                                      testbb);

      //br i1 %test.%d, label %main.%d, label %main.%d
      BasicBlock *bb_0 = BasicBlock::Create(label, brainf_func);
      BranchInst::Create(bb_0, oldbb, test_0, testbb);

      //main.%d:
      builder->SetInsertPoint(bb_0);

      //%head.%d = phi i8 *[%head.%d, %main.%d]
      PHINode *phi_1 = builder->
        CreatePHI(PointerType::getUnqual(IntegerType::Int8Ty), headreg);
      phi_1->reserveOperandSpace(1);
      phi_1->addIncoming(head_0, testbb);
      curhead = phi_1;
    }

    return;
  }

  //End of the program, so go to return block
  builder->CreateBr(endbb);

  if (phi) {
    std::cerr << "Error: Missing ']'\n";
    abort();
  }
}
