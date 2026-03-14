; RUN: opt -mtriple=x86_64-unknown-linux-gnu -passes=wholeprogramdevirt -whole-program-visibility -wholeprogramdevirt-summary-action=export -wholeprogramdevirt-read-summary=%S/Inputs/export.yaml -wholeprogramdevirt-write-summary=%t -S -o - %s | FileCheck --check-prefixes=CHECK,X86 %s
; RUN: FileCheck --check-prefixes=SUMMARY,SUMMARY-X86 %s < %t

; RUN: opt -mtriple=armv7-unknown-linux-gnu -passes=wholeprogramdevirt -whole-program-visibility -wholeprogramdevirt-summary-action=export -wholeprogramdevirt-read-summary=%S/Inputs/export.yaml -wholeprogramdevirt-write-summary=%t -S -o - %s | FileCheck --check-prefixes=CHECK,ARM %s
; RUN: FileCheck --check-prefixes=SUMMARY,SUMMARY-ARM %s < %t

target datalayout = "e-p:64:64"

; SUMMARY:      TypeIdMap:
; SUMMARY-NEXT:   typeid3:
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
; SUMMARY-NEXT:           12,24:
; SUMMARY-NEXT:             Kind:            VirtualConstProp
; SUMMARY-NEXT:             Info:            0
; SUMMARY-X86-NEXT:         Byte:            0
; SUMMARY-X86-NEXT:         Bit:             0
; SUMMARY-ARM-NEXT:         Byte:            4294967295
; SUMMARY-ARM-NEXT:         Bit:             1
; SUMMARY-NEXT:   typeid4:
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
; SUMMARY-NEXT:           24,12:
; SUMMARY-NEXT:             Kind:            VirtualConstProp
; SUMMARY-NEXT:             Info:            0
; SUMMARY-X86-NEXT:         Byte:            0
; SUMMARY-X86-NEXT:         Bit:             0
; SUMMARY-ARM-NEXT:         Byte:            4294967292
; SUMMARY-ARM-NEXT:         Bit:             1

; CHECK: [[CVT3A:.*]] = private constant { [8 x i8], ptr, [0 x i8] } { [8 x i8] zeroinitializer, ptr @vf0i1, [0 x i8] zeroinitializer }, !type !0
@vt3a = constant ptr @vf0i1, !type !0

; CHECK: [[CVT3B:.*]] = private constant { [8 x i8], ptr, [0 x i8] } { [8 x i8] c"\00\00\00\00\00\00\00\01", ptr @vf1i1, [0 x i8] zeroinitializer }, !type !0
@vt3b = constant ptr @vf1i1, !type !0

; CHECK: [[CVT3C:.*]] = private constant { [8 x i8], ptr, [0 x i8] } { [8 x i8] zeroinitializer, ptr @vf0i1, [0 x i8] zeroinitializer }, !type !0
@vt3c = constant ptr @vf0i1, !type !0

; CHECK: [[CVT3D:.*]] = private constant { [8 x i8], ptr, [0 x i8] } { [8 x i8] c"\00\00\00\00\00\00\00\01", ptr @vf1i1, [0 x i8] zeroinitializer }, !type !0
@vt3d = constant ptr @vf1i1, !type !0

; CHECK: [[CVT4A:.*]] = private constant { [8 x i8], ptr, [0 x i8] } { [8 x i8] c"\00\00\00\00\01\00\00\00", ptr @vf1i32, [0 x i8] zeroinitializer }, !type !1
@vt4a = constant ptr @vf1i32, !type !1

; CHECK: [[CVT4B:.*]] = private constant { [8 x i8], ptr, [0 x i8] } { [8 x i8] c"\00\00\00\00\02\00\00\00", ptr @vf2i32, [0 x i8] zeroinitializer }, !type !1
@vt4b = constant ptr @vf2i32, !type !1

; X86: @__typeid_typeid3_0_12_24_byte = hidden alias i8, inttoptr (i32 -1 to ptr)
; X86: @__typeid_typeid3_0_12_24_bit = hidden alias i8, inttoptr (i32 1 to ptr)
; X86: @__typeid_typeid4_0_24_12_byte = hidden alias i8, inttoptr (i32 -4 to ptr)
; X86: @__typeid_typeid4_0_24_12_bit = hidden alias i8, inttoptr (i32 1 to ptr)
; ARM-NOT: alias {{.*}} inttoptr

; CHECK: @vt3a = alias ptr, getelementptr inbounds ({ [8 x i8], ptr, [0 x i8] }, ptr [[CVT3A]], i32 0, i32 1)
; CHECK: @vt3b = alias ptr, getelementptr inbounds ({ [8 x i8], ptr, [0 x i8] }, ptr [[CVT3B]], i32 0, i32 1)
; CHECK: @vt3c = alias ptr, getelementptr inbounds ({ [8 x i8], ptr, [0 x i8] }, ptr [[CVT3C]], i32 0, i32 1)
; CHECK: @vt3d = alias ptr, getelementptr inbounds ({ [8 x i8], ptr, [0 x i8] }, ptr [[CVT3D]], i32 0, i32 1)
; CHECK: @vt4a = alias ptr, getelementptr inbounds ({ [8 x i8], ptr, [0 x i8] }, ptr [[CVT4A]], i32 0, i32 1)
; CHECK: @vt4b = alias ptr, getelementptr inbounds ({ [8 x i8], ptr, [0 x i8] }, ptr [[CVT4B]], i32 0, i32 1)

define i1 @vf0i1(ptr %this, i32, i32) readnone {
  ret i1 0
}

define i1 @vf1i1(ptr %this, i32, i32) readnone {
  ret i1 1
}

define i32 @vf1i32(ptr %this, i32, i32) readnone {
  ret i32 1
}

define i32 @vf2i32(ptr %this, i32, i32) readnone {
  ret i32 2
}

; CHECK: !0 = !{i32 8, !"typeid3"}
; CHECK: !1 = !{i32 8, !"typeid4"}

!0 = !{i32 0, !"typeid3"}
!1 = !{i32 0, !"typeid4"}
