; RUN: opt -module-summary %s -o %t.o
; RUN: llvm-bcanalyzer -dump %t.o | FileCheck %s
; RUN: llvm-lto -thinlto -o %t2 %t.o
; RUN: llvm-bcanalyzer -dump %t2.thinlto.bc | FileCheck --check-prefix=COMBINED %s

target datalayout = "e-p:64:64"
target triple = "x86_64-unknown-linux-gnu"

; COMBINED:      <TYPE_TEST_ASSUME_VCALLS op0=6699318081062747564 op1=16/>
; COMBINED-NEXT: <COMBINED
; COMBINED-NEXT: <TYPE_CHECKED_LOAD_VCALLS op0=6699318081062747564 op1=16/>
; COMBINED-NEXT: <COMBINED
; COMBINED-NEXT: <TYPE_TEST_ASSUME_VCALLS op0=6699318081062747564 op1=24 op2=-2012135647395072713 op3=32/>
; COMBINED-NEXT: <COMBINED
; COMBINED-NEXT: <TYPE_TEST_ASSUME_CONST_VCALL op0=6699318081062747564 op1=16 op2=42/>
; COMBINED-NEXT: <TYPE_TEST_ASSUME_CONST_VCALL op0=6699318081062747564 op1=24 op2=43/>
; COMBINED-NEXT: <COMBINED
; COMBINED-NEXT: <TYPE_CHECKED_LOAD_CONST_VCALL op0=6699318081062747564 op1=16 op2=42/>
; COMBINED-NEXT: <COMBINED
; COMBINED-NEXT: <TYPE_TESTS op0=7546896869197086323/>
; COMBINED-NEXT: <COMBINED

; CHECK: <TYPE_TEST_ASSUME_VCALLS op0=6699318081062747564 op1=16/>
define void @f1([3 x i8*]* %vtable) {
  %vtablei8 = bitcast [3 x i8*]* %vtable to i8*
  %p = call i1 @llvm.type.test(i8* %vtablei8, metadata !"foo")
  call void @llvm.assume(i1 %p)
  %fptrptr = getelementptr [3 x i8*], [3 x i8*]* %vtable, i32 0, i32 2
  %fptr = load i8*, i8** %fptrptr
  %fptr_casted = bitcast i8* %fptr to void (i8*, i32)*
  call void %fptr_casted(i8* null, i32 undef)
  ret void
}

; CHECK: <TYPE_TEST_ASSUME_VCALLS op0=6699318081062747564 op1=24 op2=-2012135647395072713 op3=32/>
define void @f2([3 x i8*]* %vtable, [3 x i8*]* %vtable2) {
  %vtablei8 = bitcast [3 x i8*]* %vtable to i8*
  %p = call i1 @llvm.type.test(i8* %vtablei8, metadata !"foo")
  call void @llvm.assume(i1 %p)
  %fptrptr = getelementptr [3 x i8*], [3 x i8*]* %vtable, i32 0, i32 3
  %fptr = load i8*, i8** %fptrptr
  %fptr_casted = bitcast i8* %fptr to void (i8*, i32)*
  call void %fptr_casted(i8* null, i32 undef)

  %vtablei82 = bitcast [3 x i8*]* %vtable2 to i8*
  %p2 = call i1 @llvm.type.test(i8* %vtablei82, metadata !"bar")
  call void @llvm.assume(i1 %p2)
  %fptrptr2 = getelementptr [3 x i8*], [3 x i8*]* %vtable2, i32 0, i32 4
  %fptr2 = load i8*, i8** %fptrptr2
  %fptr_casted2 = bitcast i8* %fptr2 to void (i8*, i128)*
  call void %fptr_casted2(i8* null, i128 0)

  ret void
}

; CHECK: <TYPE_CHECKED_LOAD_VCALLS op0=6699318081062747564 op1=16/>
define void @f3(i8* %vtable) {
  %pair = call {i8*, i1} @llvm.type.checked.load(i8* %vtable, i32 16, metadata !"foo")
  %fptr = extractvalue {i8*, i1} %pair, 0
  %fptr_casted = bitcast i8* %fptr to void (i8*, i32)*
  call void %fptr_casted(i8* null, i32 undef)
  ret void
}

; CHECK: <TYPE_TEST_ASSUME_CONST_VCALL op0=6699318081062747564 op1=16 op2=42/>
; CHECK-NEXT: <TYPE_TEST_ASSUME_CONST_VCALL op0=6699318081062747564 op1=24 op2=43/>
define void @f4([3 x i8*]* %vtable, [3 x i8*]* %vtable2) {
  %vtablei8 = bitcast [3 x i8*]* %vtable to i8*
  %p = call i1 @llvm.type.test(i8* %vtablei8, metadata !"foo")
  call void @llvm.assume(i1 %p)
  %fptrptr = getelementptr [3 x i8*], [3 x i8*]* %vtable, i32 0, i32 2
  %fptr = load i8*, i8** %fptrptr
  %fptr_casted = bitcast i8* %fptr to void (i8*, i32)*
  call void %fptr_casted(i8* null, i32 42)

  %vtablei82 = bitcast [3 x i8*]* %vtable2 to i8*
  %p2 = call i1 @llvm.type.test(i8* %vtablei82, metadata !"foo")
  call void @llvm.assume(i1 %p2)
  %fptrptr2 = getelementptr [3 x i8*], [3 x i8*]* %vtable2, i32 0, i32 3
  %fptr2 = load i8*, i8** %fptrptr2
  %fptr_casted2 = bitcast i8* %fptr2 to void (i8*, i32)*
  call void %fptr_casted2(i8* null, i32 43)
  ret void
}

; CHECK: <TYPE_CHECKED_LOAD_CONST_VCALL op0=6699318081062747564 op1=16 op2=42/>
define void @f5(i8* %vtable) {
  %pair = call {i8*, i1} @llvm.type.checked.load(i8* %vtable, i32 16, metadata !"foo")
  %fptr = extractvalue {i8*, i1} %pair, 0
  %fptr_casted = bitcast i8* %fptr to void (i8*, i32)*
  call void %fptr_casted(i8* null, i32 42)
  ret void
}

; CHECK-NOT: <TYPE_CHECKED_LOAD_CONST_VCALL op0=7546896869197086323
; CHECK: <TYPE_TESTS op0=7546896869197086323/>
; CHECK-NOT: <TYPE_CHECKED_LOAD_CONST_VCALL op0=7546896869197086323
define {i8*, i1} @f6(i8* %vtable) {
  %pair = call {i8*, i1} @llvm.type.checked.load(i8* %vtable, i32 16, metadata !"baz")
  ret {i8*, i1} %pair
}

declare i1 @llvm.type.test(i8*, metadata) nounwind readnone
declare void @llvm.assume(i1)
declare {i8*, i1} @llvm.type.checked.load(i8*, i32, metadata)
