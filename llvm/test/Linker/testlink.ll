; RUN: llvm-link %s %S/Inputs/testlink.ll -S | FileCheck %s

; CHECK: %Ty1 = type { %Ty2* }
; CHECK: %Ty2 = type { %Ty1* }
%Ty1 = type opaque
%Ty2 = type { %Ty1* }

; CHECK: %intlist = type { %intlist*, i32 }
%intlist = type { %intlist*, i32 }

; The uses of intlist in the other file should be remapped.
; CHECK-NOT: {{%intlist.[0-9]}}

; CHECK: %VecSize = type { <5 x i32> }
; CHECK: %VecSize.{{[0-9]}} = type { <10 x i32> }
%VecSize = type { <5 x i32> }

%Struct1 = type opaque
@S1GV = external global %Struct1*


@GVTy1 = external global %Ty1*
@GVTy2 = global %Ty2* null


; This should stay the same
; CHECK-DAG: @MyIntList = global %intlist { %intlist* null, i32 17 }
@MyIntList = global %intlist { %intlist* null, i32 17 }


; Nothing to link here.

; CHECK-DAG: @0 = external global i32
@0 = external global i32

define i32* @use0() {
  ret i32* @0
}

; CHECK-DAG: @Inte = global i32 1
@Inte = global i32 1

; Intern1 is intern in both files, rename testlink2's.
; CHECK-DAG: @Intern1 = internal constant i32 42
@Intern1 = internal constant i32 42

@UseIntern1 = global i32* @Intern1

; This should get renamed since there is a definition that is non-internal in
; the other module.
; CHECK-DAG: @Intern2.{{[0-9]+}} = internal constant i32 792
@Intern2 = internal constant i32 792

@UseIntern2 = global i32* @Intern2

; CHECK-DAG: @MyVarPtr = linkonce global { i32* } { i32* @MyVar }
@MyVarPtr = linkonce global { i32* } { i32* @MyVar }

@UseMyVarPtr = global { i32* }* @MyVarPtr

; CHECK-DAG: @MyVar = global i32 4
@MyVar = external global i32

; Take value from other module.
; CHECK-DAG: AConst = constant i32 1234
@AConst = linkonce constant i32 123

; Renamed version of Intern1.
; CHECK-DAG: @Intern1.{{[0-9]+}} = internal constant i32 52


; Globals linked from testlink2.
; CHECK-DAG: @Intern2 = constant i32 12345

; CHECK-DAG: @MyIntListPtr = constant
; CHECK-DAG: @1 = constant i32 412


declare i32 @foo(i32)

declare void @print(i32)

define void @main() {
  %v1 = load i32, i32* @MyVar
  call void @print(i32 %v1)
  %idx = getelementptr %intlist, %intlist* @MyIntList, i64 0, i32 1
  %v2 = load i32, i32* %idx
  call void @print(i32 %v2)
  %1 = call i32 @foo(i32 5)
  %v3 = load i32, i32* @MyVar
  call void @print(i32 %v3)
  %v4 = load i32, i32* %idx
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

define void @VecSizeCrash(%VecSize) {
  ret void
}
