; Test that we correctly import an indir resolution for type identifier "typeid1".
; RUN: opt -S -wholeprogramdevirt -wholeprogramdevirt-summary-action=import -wholeprogramdevirt-read-summary=%S/Inputs/import-indir.yaml -wholeprogramdevirt-write-summary=%t < %s | FileCheck %s
; RUN: FileCheck --check-prefix=SUMMARY %s < %t

; SUMMARY:     GlobalValueMap:
; SUMMARY-NEXT:  42:
; SUMMARY-NEXT:    - TypeTests:
; SUMMARY-NEXT:      TypeTestAssumeVCalls:
; SUMMARY-NEXT:        - GUID:            123
; SUMMARY-NEXT:          Offset:          0
; SUMMARY-NEXT:        - GUID:            456
; SUMMARY-NEXT:          Offset:          4
; SUMMARY-NEXT:      TypeCheckedLoadVCalls:
; SUMMARY-NEXT:        - GUID:            789
; SUMMARY-NEXT:          Offset:          8
; SUMMARY-NEXT:        - GUID:            1234
; SUMMARY-NEXT:          Offset:          16
; SUMMARY-NEXT:      TypeTestAssumeConstVCalls:
; SUMMARY-NEXT:        - VFunc:
; SUMMARY-NEXT:            GUID:            123
; SUMMARY-NEXT:            Offset:          4
; SUMMARY-NEXT:          Args:
; SUMMARY-NEXT:            - 12
; SUMMARY-NEXT:            - 24
; SUMMARY-NEXT:      TypeCheckedLoadConstVCalls:
; SUMMARY-NEXT:        - VFunc:
; SUMMARY-NEXT:            GUID:            456
; SUMMARY-NEXT:            Offset:          8
; SUMMARY-NEXT:          Args:
; SUMMARY-NEXT:            - 24
; SUMMARY-NEXT:            - 12
; SUMMARY-NEXT: TypeIdMap:
; SUMMARY-NEXT:   typeid1:
; SUMMARY-NEXT:     TTRes:
; SUMMARY-NEXT:       Kind:            Unsat
; SUMMARY-NEXT:       SizeM1BitWidth:  0
; SUMMARY-NEXT:     WPDRes:
; SUMMARY-NEXT:       0:
; SUMMARY-NEXT:         Kind:            Indir
; SUMMARY-NEXT:         SingleImplName:  ''
; SUMMARY-NEXT:         ResByArg:
; SUMMARY-NEXT:       4:
; SUMMARY-NEXT:         Kind:            Indir
; SUMMARY-NEXT:         SingleImplName:  ''
; SUMMARY-NEXT:         ResByArg:
; SUMMARY-NEXT:           :
; SUMMARY-NEXT:             Kind:            UniformRetVal
; SUMMARY-NEXT:             Info:            12
; SUMMARY-NEXT:           12:
; SUMMARY-NEXT:             Kind:            UniformRetVal
; SUMMARY-NEXT:             Info:            24
; SUMMARY-NEXT:           12,24:
; SUMMARY-NEXT:             Kind:            UniformRetVal
; SUMMARY-NEXT:             Info:            48

target datalayout = "e-p:32:32"

declare void @llvm.assume(i1)
declare void @llvm.trap()
declare {i8*, i1} @llvm.type.checked.load(i8*, i32, metadata)
declare i1 @llvm.type.test(i8*, metadata)

; CHECK: define i1 @f1
define i1 @f1(i8* %obj) {
  %vtableptr = bitcast i8* %obj to [1 x i8*]**
  %vtable = load [1 x i8*]*, [1 x i8*]** %vtableptr
  %vtablei8 = bitcast [1 x i8*]* %vtable to i8*
  %p = call i1 @llvm.type.test(i8* %vtablei8, metadata !"typeid1")
  call void @llvm.assume(i1 %p)
  %fptrptr = getelementptr [1 x i8*], [1 x i8*]* %vtable, i32 0, i32 0
  %fptr = load i8*, i8** %fptrptr
  %fptr_casted = bitcast i8* %fptr to i1 (i8*, i32)*
  ; CHECK: call i1 %
  %result = call i1 %fptr_casted(i8* %obj, i32 5)
  ret i1 %result
}

; CHECK: define i1 @f2
define i1 @f2(i8* %obj) {
  %vtableptr = bitcast i8* %obj to [1 x i8*]**
  %vtable = load [1 x i8*]*, [1 x i8*]** %vtableptr
  %vtablei8 = bitcast [1 x i8*]* %vtable to i8*
  %pair = call {i8*, i1} @llvm.type.checked.load(i8* %vtablei8, i32 4, metadata !"typeid1")
  %fptr = extractvalue {i8*, i1} %pair, 0
  %p = extractvalue {i8*, i1} %pair, 1
  ; CHECK: [[P:%.*]] = call i1 @llvm.type.test
  ; CHECK: br i1 [[P]]
  br i1 %p, label %cont, label %trap

cont:
  %fptr_casted = bitcast i8* %fptr to i1 (i8*, i32)*
  ; CHECK: call i1 %
  %result = call i1 %fptr_casted(i8* %obj, i32 undef)
  ret i1 %result

trap:
  call void @llvm.trap()
  unreachable
}
