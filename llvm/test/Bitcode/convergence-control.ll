; RUN: llvm-dis < %s.bc | FileCheck %s

define void @loop_nesting() convergent {
A:
  ; CHECK-LABEL: A:
  ; CHECK: [[A:%.*]] = call token @llvm.experimental.convergence.entry()
  ;
  %a = call token @llvm.experimental.convergence.entry()
  br label %B

B:
  ; CHECK-LABEL: B:
  ; CHECK: [[B:%.*]] = call token @llvm.experimental.convergence.anchor()
  ;
  %b = call token @llvm.experimental.convergence.anchor()
  br i1 undef, label %C, label %D

C:
  ; CHECK-LABEL: C:
  ; CHECK: [[C:%.*]] = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token [[B]]) ]
  ; CHEC K: call void @f() [ "convergencectrl"(token [[C]]) ]
  ;
  %c = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token %b) ]
  call void @f() [ "convergencectrl"(token %c) ]
  br label %B

D:
  ; CHECK-LABEL: D:
  ; CHECK:  call void @f() [ "convergencectrl"(token [[B]]) ]
  ;
  call void @f() [ "convergencectrl"(token %b) ]
  br i1 undef, label %B, label %E

E:
  ret void
}

declare void @f() convergent

declare token @llvm.experimental.convergence.entry()
declare token @llvm.experimental.convergence.anchor()
declare token @llvm.experimental.convergence.loop()
