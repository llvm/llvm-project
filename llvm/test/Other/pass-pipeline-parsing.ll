; RUN: opt -disable-output -debug-pass-manager \
; RUN:     -passes=no-op-module,no-op-module %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-TWO-NOOP-MP
; CHECK-TWO-NOOP-MP: Starting llvm::Module pass manager run
; CHECK-TWO-NOOP-MP: Running pass: NoOpModulePass
; CHECK-TWO-NOOP-MP: Running pass: NoOpModulePass
; CHECK-TWO-NOOP-MP: Finished llvm::Module pass manager run

; RUN: opt -disable-output -debug-pass-manager \
; RUN:     -passes='module(no-op-module,no-op-module)' %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-NESTED-TWO-NOOP-MP
; CHECK-NESTED-TWO-NOOP-MP: Starting llvm::Module pass manager run
; CHECK-NESTED-TWO-NOOP-MP: Starting llvm::Module pass manager run
; CHECK-NESTED-TWO-NOOP-MP: Running pass: NoOpModulePass
; CHECK-NESTED-TWO-NOOP-MP: Running pass: NoOpModulePass
; CHECK-NESTED-TWO-NOOP-MP: Finished llvm::Module pass manager run
; CHECK-NESTED-TWO-NOOP-MP: Finished llvm::Module pass manager run

; RUN: opt -disable-output -debug-pass-manager \
; RUN:     -passes=no-op-function,no-op-function %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-TWO-NOOP-FP
; CHECK-TWO-NOOP-FP: Starting llvm::Module pass manager run
; CHECK-TWO-NOOP-FP: Running pass: ModuleToFunctionPassAdaptor
; CHECK-TWO-NOOP-FP: Starting llvm::Function pass manager run
; CHECK-TWO-NOOP-FP: Running pass: NoOpFunctionPass
; CHECK-TWO-NOOP-FP: Running pass: NoOpFunctionPass
; CHECK-TWO-NOOP-FP: Finished llvm::Function pass manager run
; CHECK-TWO-NOOP-FP: Finished llvm::Module pass manager run

; RUN: opt -disable-output -debug-pass-manager \
; RUN:     -passes='function(no-op-function,no-op-function)' %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-NESTED-TWO-NOOP-FP
; CHECK-NESTED-TWO-NOOP-FP: Starting llvm::Module pass manager run
; CHECK-NESTED-TWO-NOOP-FP: Running pass: ModuleToFunctionPassAdaptor
; CHECK-NESTED-TWO-NOOP-FP: Starting llvm::Function pass manager run
; CHECK-NESTED-TWO-NOOP-FP: Running pass: NoOpFunctionPass
; CHECK-NESTED-TWO-NOOP-FP: Running pass: NoOpFunctionPass
; CHECK-NESTED-TWO-NOOP-FP: Finished llvm::Function pass manager run
; CHECK-NESTED-TWO-NOOP-FP: Finished llvm::Module pass manager run

; RUN: opt -disable-output -debug-pass-manager \
; RUN:     -passes='no-op-module,function(no-op-function,no-op-function),no-op-module' %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-MIXED-FP-AND-MP
; CHECK-MIXED-FP-AND-MP: Starting llvm::Module pass manager run
; CHECK-MIXED-FP-AND-MP: Running pass: NoOpModulePass
; CHECK-MIXED-FP-AND-MP: Running pass: ModuleToFunctionPassAdaptor
; CHECK-MIXED-FP-AND-MP: Starting llvm::Function pass manager run
; CHECK-MIXED-FP-AND-MP: Running pass: NoOpFunctionPass
; CHECK-MIXED-FP-AND-MP: Running pass: NoOpFunctionPass
; CHECK-MIXED-FP-AND-MP: Finished llvm::Function pass manager run
; CHECK-MIXED-FP-AND-MP: Running pass: NoOpModulePass
; CHECK-MIXED-FP-AND-MP: Finished llvm::Module pass manager run

; RUN: not opt -disable-output -debug-pass-manager \
; RUN:     -passes='no-op-module)' %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-UNBALANCED1
; CHECK-UNBALANCED1: unable to parse pass pipeline description

; RUN: not opt -disable-output -debug-pass-manager \
; RUN:     -passes='module(no-op-module))' %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-UNBALANCED2
; CHECK-UNBALANCED2: unable to parse pass pipeline description

; RUN: not opt -disable-output -debug-pass-manager \
; RUN:     -passes='module(no-op-module' %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-UNBALANCED3
; CHECK-UNBALANCED3: unable to parse pass pipeline description

; RUN: not opt -disable-output -debug-pass-manager \
; RUN:     -passes='no-op-function)' %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-UNBALANCED4
; CHECK-UNBALANCED4: unable to parse pass pipeline description

; RUN: not opt -disable-output -debug-pass-manager \
; RUN:     -passes='function(no-op-function))' %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-UNBALANCED5
; CHECK-UNBALANCED5: unable to parse pass pipeline description

; RUN: not opt -disable-output -debug-pass-manager \
; RUN:     -passes='function(function(no-op-function)))' %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-UNBALANCED6
; CHECK-UNBALANCED6: unable to parse pass pipeline description

; RUN: not opt -disable-output -debug-pass-manager \
; RUN:     -passes='function(no-op-function' %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-UNBALANCED7
; CHECK-UNBALANCED7: unable to parse pass pipeline description

; RUN: not opt -disable-output -debug-pass-manager \
; RUN:     -passes='function(function(no-op-function)' %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-UNBALANCED8
; CHECK-UNBALANCED8: unable to parse pass pipeline description

; RUN: not opt -disable-output -debug-pass-manager \
; RUN:     -passes='no-op-module,)' %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-UNBALANCED9
; CHECK-UNBALANCED9: unable to parse pass pipeline description

; RUN: not opt -disable-output -debug-pass-manager \
; RUN:     -passes='no-op-function,)' %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-UNBALANCED10
; CHECK-UNBALANCED10: unable to parse pass pipeline description

; RUN: opt -disable-output -debug-pass-manager \
; RUN:     -passes=no-op-cgscc,no-op-cgscc %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-TWO-NOOP-CG
; CHECK-TWO-NOOP-CG: Starting llvm::Module pass manager run
; CHECK-TWO-NOOP-CG: Running pass: ModuleToPostOrderCGSCCPassAdaptor
; CHECK-TWO-NOOP-CG: Starting llvm::LazyCallGraph::SCC pass manager run
; CHECK-TWO-NOOP-CG: Running pass: NoOpCGSCCPass
; CHECK-TWO-NOOP-CG: Running pass: NoOpCGSCCPass
; CHECK-TWO-NOOP-CG: Finished llvm::LazyCallGraph::SCC pass manager run
; CHECK-TWO-NOOP-CG: Finished llvm::Module pass manager run

; RUN: opt -disable-output -debug-pass-manager \
; RUN:     -passes='module(function(no-op-function),cgscc(no-op-cgscc,function(no-op-function),no-op-cgscc),function(no-op-function))' %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-NESTED-MP-CG-FP
; CHECK-NESTED-MP-CG-FP: Starting llvm::Module pass manager run
; CHECK-NESTED-MP-CG-FP: Starting llvm::Module pass manager run
; CHECK-NESTED-MP-CG-FP: Running pass: ModuleToFunctionPassAdaptor
; CHECK-NESTED-MP-CG-FP: Starting llvm::Function pass manager run
; CHECK-NESTED-MP-CG-FP: Running pass: NoOpFunctionPass
; CHECK-NESTED-MP-CG-FP: Finished llvm::Function pass manager run
; CHECK-NESTED-MP-CG-FP: Running pass: ModuleToPostOrderCGSCCPassAdaptor
; CHECK-NESTED-MP-CG-FP: Starting llvm::LazyCallGraph::SCC pass manager run
; CHECK-NESTED-MP-CG-FP: Running pass: NoOpCGSCCPass
; CHECK-NESTED-MP-CG-FP: Running pass: CGSCCToFunctionPassAdaptor
; CHECK-NESTED-MP-CG-FP: Starting llvm::Function pass manager run
; CHECK-NESTED-MP-CG-FP: Running pass: NoOpFunctionPass
; CHECK-NESTED-MP-CG-FP: Finished llvm::Function pass manager run
; CHECK-NESTED-MP-CG-FP: Running pass: NoOpCGSCCPass
; CHECK-NESTED-MP-CG-FP: Finished llvm::LazyCallGraph::SCC pass manager run
; CHECK-NESTED-MP-CG-FP: Running pass: ModuleToFunctionPassAdaptor
; CHECK-NESTED-MP-CG-FP: Starting llvm::Function pass manager run
; CHECK-NESTED-MP-CG-FP: Running pass: NoOpFunctionPass
; CHECK-NESTED-MP-CG-FP: Finished llvm::Function pass manager run
; CHECK-NESTED-MP-CG-FP: Finished llvm::Module pass manager run
; CHECK-NESTED-MP-CG-FP: Finished llvm::Module pass manager run

; RUN: opt -disable-output -debug-pass-manager \
; RUN:     -passes='no-op-loop,no-op-loop' %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-TWO-NOOP-LOOP
; CHECK-TWO-NOOP-LOOP: Starting llvm::Module pass manager run
; CHECK-TWO-NOOP-LOOP: Running pass: ModuleToFunctionPassAdaptor
; CHECK-TWO-NOOP-LOOP: Starting llvm::Function pass manager run
; CHECK-TWO-NOOP-LOOP: Running pass: FunctionToLoopPassAdaptor
; CHECK-TWO-NOOP-LOOP: Starting llvm::Loop pass manager run
; CHECK-TWO-NOOP-LOOP: Running pass: NoOpLoopPass
; CHECK-TWO-NOOP-LOOP: Running pass: NoOpLoopPass
; CHECK-TWO-NOOP-LOOP: Finished llvm::Loop pass manager run
; CHECK-TWO-NOOP-LOOP: Finished llvm::Function pass manager run
; CHECK-TWO-NOOP-LOOP: Finished llvm::Module pass manager run

; RUN: opt -disable-output -debug-pass-manager \
; RUN:     -passes='module(function(loop(no-op-loop)))' %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-NESTED-FP-LP
; RUN: opt -disable-output -debug-pass-manager \
; RUN:     -passes='function(loop(no-op-loop))' %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-NESTED-FP-LP
; RUN: opt -disable-output -debug-pass-manager \
; RUN:     -passes='loop(no-op-loop)' %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-NESTED-FP-LP
; RUN: opt -disable-output -debug-pass-manager \
; RUN:     -passes='no-op-loop' %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-NESTED-FP-LP
; CHECK-NESTED-FP-LP: Starting llvm::Module pass manager run
; CHECK-NESTED-FP-LP: Running pass: ModuleToFunctionPassAdaptor
; CHECK-NESTED-FP-LP: Starting llvm::Function pass manager run
; CHECK-NESTED-FP-LP: Running pass: FunctionToLoopPassAdaptor
; CHECK-NESTED-FP-LP: Starting llvm::Loop pass manager run
; CHECK-NESTED-FP-LP: Running pass: NoOpLoopPass
; CHECK-NESTED-FP-LP: Finished llvm::Loop pass manager run
; CHECK-NESTED-FP-LP: Finished llvm::Function pass manager run
; CHECK-NESTED-FP-LP: Finished llvm::Module pass manager run

; RUN: not opt -disable-output -debug-pass-manager \
; RUN:     -passes='function(no-op-function)function(no-op-function)' %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-MISSING-COMMA1
; CHECK-MISSING-COMMA1: unable to parse pass pipeline description

; RUN: not opt -disable-output -debug-pass-manager \
; RUN:     -passes='function()' %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-EMPTY-INNER-PIPELINE
; CHECK-EMPTY-INNER-PIPELINE: unable to parse pass pipeline description

; RUN: not opt -disable-output -debug-pass-manager \
; RUN:     -passes='no-op-module(no-op-module,whatever)' %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-PIPELINE-ON-MODULE-PASS
; CHECK-PIPELINE-ON-MODULE-PASS: unable to parse pass pipeline description

; RUN: not opt -disable-output -debug-pass-manager \
; RUN:     -passes='no-op-cgscc(no-op-cgscc,whatever)' %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-PIPELINE-ON-CGSCC-PASS
; CHECK-PIPELINE-ON-CGSCC-PASS: unable to parse pass pipeline description

; RUN: not opt -disable-output -debug-pass-manager \
; RUN:     -passes='no-op-function(no-op-function,whatever)' %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-PIPELINE-ON-FUNCTION-PASS
; CHECK-PIPELINE-ON-FUNCTION-PASS: unable to parse pass pipeline description

; RUN: not opt -disable-output -debug-pass-manager \
; RUN:     -passes='no-op-loop(no-op-loop,whatever)' %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-PIPELINE-ON-LOOP-PASS
; CHECK-PIPELINE-ON-LOOP-PASS: unable to parse pass pipeline description

; RUN: not opt -disable-output -debug-pass-manager \
; RUN:     -passes='no-op-function()' %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=CHECK-EMPTY-PIPELINE-ON-PASS
; CHECK-EMPTY-PIPELINE-ON-PASS: unable to parse pass pipeline description

define void @f() {
entry:
 br label %loop
loop:
 br label %loop
}
