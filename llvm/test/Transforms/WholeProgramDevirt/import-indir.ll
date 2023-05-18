; Test that we correctly import an indir resolution for type identifier "typeid1".
; RUN: opt -S -passes=wholeprogramdevirt -wholeprogramdevirt-summary-action=import -wholeprogramdevirt-read-summary=%S/Inputs/import-indir.yaml -wholeprogramdevirt-write-summary=%t < %s | FileCheck %s
; RUN: FileCheck --check-prefix=SUMMARY %s < %t

; SUMMARY:     GlobalValueMap:
; SUMMARY-NEXT:  42:
; SUMMARY-NEXT:    - Linkage:             0
; SUMMARY-NEXT:      Visibility:          0
; SUMMARY-NEXT:      NotEligibleToImport: false
; SUMMARY-NEXT:      Live:                true
; SUMMARY-NEXT:      Local:               false
; SUMMARY-NEXT:      CanAutoHide:         false
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
; SUMMARY-NEXT:          Args: [ 12, 24 ]
; SUMMARY-NEXT:      TypeCheckedLoadConstVCalls:
; SUMMARY-NEXT:        - VFunc:
; SUMMARY-NEXT:            GUID:            456
; SUMMARY-NEXT:            Offset:          8
; SUMMARY-NEXT:          Args: [ 24, 12 ]
; SUMMARY-NEXT: TypeIdMap:
; SUMMARY-NEXT:   typeid1:
; SUMMARY-NEXT:     TTRes:
; SUMMARY-NEXT:       Kind:            Unknown
; SUMMARY-NEXT:       SizeM1BitWidth:  0
; SUMMARY-NEXT:       AlignLog2:       0
; SUMMARY-NEXT:       SizeM1:          0
; SUMMARY-NEXT:       BitMask:         0
; SUMMARY-NEXT:       InlineBits:      0
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
; SUMMARY-NEXT:             Byte:            0
; SUMMARY-NEXT:             Bit:             0
; SUMMARY-NEXT:           12:
; SUMMARY-NEXT:             Kind:            UniformRetVal
; SUMMARY-NEXT:             Info:            24
; SUMMARY-NEXT:             Byte:            0
; SUMMARY-NEXT:             Bit:             0
; SUMMARY-NEXT:           12,24:
; SUMMARY-NEXT:             Kind:            UniformRetVal
; SUMMARY-NEXT:             Info:            48
; SUMMARY-NEXT:             Byte:            0
; SUMMARY-NEXT:             Bit:             0

target datalayout = "e-p:32:32"

declare void @llvm.assume(i1)
declare void @llvm.trap()
declare {ptr, i1} @llvm.type.checked.load(ptr, i32, metadata)
declare i1 @llvm.type.test(ptr, metadata)

; CHECK: define i1 @f1
define i1 @f1(ptr %obj) {
  %vtable = load ptr, ptr %obj
  %p = call i1 @llvm.type.test(ptr %vtable, metadata !"typeid1")
  call void @llvm.assume(i1 %p)
  %fptr = load ptr, ptr %vtable
  ; CHECK: call i1 %
  %result = call i1 %fptr(ptr %obj, i32 5)
  ret i1 %result
}

; CHECK: define i1 @f2
define i1 @f2(ptr %obj) {
  %vtable = load ptr, ptr %obj
  %pair = call {ptr, i1} @llvm.type.checked.load(ptr %vtable, i32 4, metadata !"typeid1")
  %fptr = extractvalue {ptr, i1} %pair, 0
  %p = extractvalue {ptr, i1} %pair, 1
  ; CHECK: [[P:%.*]] = call i1 @llvm.type.test
  ; CHECK: br i1 [[P]]
  br i1 %p, label %cont, label %trap

cont:
  ; CHECK: call i1 %
  %result = call i1 %fptr(ptr %obj, i32 undef)
  ret i1 %result

trap:
  call void @llvm.trap()
  unreachable
}
