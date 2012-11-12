; RUN: llvm-as < %s > %t.bc
; RUN: llvm-as < %p/testlink2.ll > %t2.bc
; RUN: llvm-link %t.bc %t2.bc -S | FileCheck %s

; CHECK: %Ty2 = type { %Ty1* }
; CHECK: %Ty1 = type { %Ty2* }
%Ty1 = type opaque
%Ty2 = type { %Ty1* }

; CHECK: %intlist = type { %intlist*, i32 }
%intlist = type { %intlist*, i32 }

; The uses of intlist in the other file should be remapped.
; CHECK-NOT: {{%intlist.[0-9]}}

%Struct1 = type opaque
@S1GV = external global %Struct1*


@GVTy1 = external global %Ty1*
@GVTy2 = global %Ty2* null


; This should stay the same
; CHECK: @MyIntList = global %intlist { %intlist* null, i32 17 }
@MyIntList = global %intlist { %intlist* null, i32 17 }


; Nothing to link here.

; CHECK: @0 = external global i32
@0 = external global i32
; CHECK: @Inte = global i32 1
@Inte = global i32 1

; Intern1 is intern in both files, rename testlink2's.
; CHECK: @Intern1 = internal constant i32 42
@Intern1 = internal constant i32 42

; This should get renamed since there is a definition that is non-internal in
; the other module.
; CHECK: @Intern2{{[0-9]+}} = internal constant i32 792
@Intern2 = internal constant i32 792


; CHECK: @MyVarPtr = linkonce global { i32* } { i32* @MyVar }
@MyVarPtr = linkonce global { i32* } { i32* @MyVar }

; CHECK: @MyVar = global i32 4
@MyVar = external global i32

; Take value from other module.
; CHECK: AConst = constant i32 1234
@AConst = linkonce constant i32 123

; Renamed version of Intern1.
; CHECK: @Intern1{{[0-9]+}} = internal constant i32 52


; Globals linked from testlink2.
; CHECK: @Intern2 = constant i32 12345

; CHECK: @MyIntListPtr = constant 
; CHECK: @1 = constant i32 412


declare i32 @foo(i32)

declare void @print(i32)

define void @main() {
  %v1 = load i32* @MyVar
  call void @print(i32 %v1)
  %idx = getelementptr %intlist* @MyIntList, i64 0, i32 1
  %v2 = load i32* %idx
  call void @print(i32 %v2)
  %1 = call i32 @foo(i32 5)
  %v3 = load i32* @MyVar
  call void @print(i32 %v3)
  %v4 = load i32* %idx
  call void @print(i32 %v4)
  ret void
}

define internal void @testintern() {
  ret void
}

define internal void @Testintern() {
  ret void
}

define void @testIntern() {
  ret void
}
