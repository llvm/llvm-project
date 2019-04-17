; RUN: opt -mtriple=x86_64-unknown-linux -S -lowertypetests -lowertypetests-summary-action=export -lowertypetests-read-summary=%S/Inputs/use-typeid1-typeid2.yaml -lowertypetests-write-summary=%t < %s | FileCheck --check-prefixes=CHECK,X86 %s
; RUN: FileCheck --check-prefixes=SUMMARY,SUMMARY-X86 %s < %t

; RUN: opt -mtriple=aarch64-unknown-linux -S -lowertypetests -lowertypetests-summary-action=export -lowertypetests-read-summary=%S/Inputs/use-typeid1-typeid2.yaml -lowertypetests-write-summary=%t < %s | FileCheck --check-prefixes=CHECK,ARM %s
; RUN: FileCheck --check-prefixes=SUMMARY,SUMMARY-ARM %s < %t

@foo = constant [2048 x i8] zeroinitializer, !type !0, !type !1, !type !2, !type !3, !type !4, !type !5, !type !6, !type !7, !type !8, !type !9, !type !10, !type !11, !type !12, !type !13, !type !14, !type !15, !type !16, !type !17, !type !18, !type !19, !type !20, !type !21, !type !22, !type !23, !type !24, !type !25, !type !26, !type !27, !type !28, !type !29, !type !30, !type !31, !type !32, !type !33, !type !34, !type !35, !type !36, !type !37, !type !38, !type !39, !type !40, !type !41, !type !42, !type !43, !type !44, !type !45, !type !46, !type !47, !type !48, !type !49, !type !50, !type !51, !type !52, !type !53, !type !54, !type !55, !type !56, !type !57, !type !58, !type !59, !type !60, !type !61, !type !62, !type !63, !type !64, !type !65, !type !66, !type !67, !type !68, !type !69, !type !70, !type !71, !type !72, !type !73, !type !74, !type !75, !type !76, !type !77, !type !78, !type !79, !type !80, !type !81, !type !82, !type !83, !type !84, !type !85, !type !86, !type !87, !type !88, !type !89, !type !90, !type !91, !type !92, !type !93, !type !94, !type !95, !type !96, !type !97, !type !98, !type !99, !type !100, !type !101, !type !102, !type !103, !type !104, !type !105, !type !106, !type !107, !type !108, !type !109, !type !110, !type !111, !type !112, !type !113, !type !114, !type !115, !type !116, !type !117, !type !118, !type !119, !type !120, !type !121, !type !122, !type !123, !type !124, !type !125, !type !126, !type !127, !type !128, !type !129, !type !130

!0 = !{i32 0, !"typeid1"}
!1 = !{i32 2, !"typeid1"}

!2 = !{i32 4, !"typeid2"}
!3 = !{i32 8, !"typeid2"}
!4 = !{i32 12, !"typeid2"}
!5 = !{i32 16, !"typeid2"}
!6 = !{i32 20, !"typeid2"}
!7 = !{i32 24, !"typeid2"}
!8 = !{i32 28, !"typeid2"}
!9 = !{i32 32, !"typeid2"}
!10 = !{i32 36, !"typeid2"}
!11 = !{i32 40, !"typeid2"}
!12 = !{i32 44, !"typeid2"}
!13 = !{i32 48, !"typeid2"}
!14 = !{i32 52, !"typeid2"}
!15 = !{i32 56, !"typeid2"}
!16 = !{i32 60, !"typeid2"}
!17 = !{i32 64, !"typeid2"}
!18 = !{i32 68, !"typeid2"}
!19 = !{i32 72, !"typeid2"}
!20 = !{i32 76, !"typeid2"}
!21 = !{i32 80, !"typeid2"}
!22 = !{i32 84, !"typeid2"}
!23 = !{i32 88, !"typeid2"}
!24 = !{i32 92, !"typeid2"}
!25 = !{i32 96, !"typeid2"}
!26 = !{i32 100, !"typeid2"}
!27 = !{i32 104, !"typeid2"}
!28 = !{i32 108, !"typeid2"}
!29 = !{i32 112, !"typeid2"}
!30 = !{i32 116, !"typeid2"}
!31 = !{i32 120, !"typeid2"}
!32 = !{i32 124, !"typeid2"}
!33 = !{i32 128, !"typeid2"}
!34 = !{i32 132, !"typeid2"}
!35 = !{i32 136, !"typeid2"}
!36 = !{i32 140, !"typeid2"}
!37 = !{i32 144, !"typeid2"}
!38 = !{i32 148, !"typeid2"}
!39 = !{i32 152, !"typeid2"}
!40 = !{i32 156, !"typeid2"}
!41 = !{i32 160, !"typeid2"}
!42 = !{i32 164, !"typeid2"}
!43 = !{i32 168, !"typeid2"}
!44 = !{i32 172, !"typeid2"}
!45 = !{i32 176, !"typeid2"}
!46 = !{i32 180, !"typeid2"}
!47 = !{i32 184, !"typeid2"}
!48 = !{i32 188, !"typeid2"}
!49 = !{i32 192, !"typeid2"}
!50 = !{i32 196, !"typeid2"}
!51 = !{i32 200, !"typeid2"}
!52 = !{i32 204, !"typeid2"}
!53 = !{i32 208, !"typeid2"}
!54 = !{i32 212, !"typeid2"}
!55 = !{i32 216, !"typeid2"}
!56 = !{i32 220, !"typeid2"}
!57 = !{i32 224, !"typeid2"}
!58 = !{i32 228, !"typeid2"}
!59 = !{i32 232, !"typeid2"}
!60 = !{i32 236, !"typeid2"}
!61 = !{i32 240, !"typeid2"}
!62 = !{i32 244, !"typeid2"}
!63 = !{i32 248, !"typeid2"}
!64 = !{i32 252, !"typeid2"}
!65 = !{i32 256, !"typeid2"}
!66 = !{i32 260, !"typeid2"}
!67 = !{i32 264, !"typeid2"}
!68 = !{i32 268, !"typeid2"}
!69 = !{i32 272, !"typeid2"}
!70 = !{i32 276, !"typeid2"}
!71 = !{i32 280, !"typeid2"}
!72 = !{i32 284, !"typeid2"}
!73 = !{i32 288, !"typeid2"}
!74 = !{i32 292, !"typeid2"}
!75 = !{i32 296, !"typeid2"}
!76 = !{i32 300, !"typeid2"}
!77 = !{i32 304, !"typeid2"}
!78 = !{i32 308, !"typeid2"}
!79 = !{i32 312, !"typeid2"}
!80 = !{i32 316, !"typeid2"}
!81 = !{i32 320, !"typeid2"}
!82 = !{i32 324, !"typeid2"}
!83 = !{i32 328, !"typeid2"}
!84 = !{i32 332, !"typeid2"}
!85 = !{i32 336, !"typeid2"}
!86 = !{i32 340, !"typeid2"}
!87 = !{i32 344, !"typeid2"}
!88 = !{i32 348, !"typeid2"}
!89 = !{i32 352, !"typeid2"}
!90 = !{i32 356, !"typeid2"}
!91 = !{i32 360, !"typeid2"}
!92 = !{i32 364, !"typeid2"}
!93 = !{i32 368, !"typeid2"}
!94 = !{i32 372, !"typeid2"}
!95 = !{i32 376, !"typeid2"}
!96 = !{i32 380, !"typeid2"}
!97 = !{i32 384, !"typeid2"}
!98 = !{i32 388, !"typeid2"}
!99 = !{i32 392, !"typeid2"}
!100 = !{i32 396, !"typeid2"}
!101 = !{i32 400, !"typeid2"}
!102 = !{i32 404, !"typeid2"}
!103 = !{i32 408, !"typeid2"}
!104 = !{i32 412, !"typeid2"}
!105 = !{i32 416, !"typeid2"}
!106 = !{i32 420, !"typeid2"}
!107 = !{i32 424, !"typeid2"}
!108 = !{i32 428, !"typeid2"}
!109 = !{i32 432, !"typeid2"}
!110 = !{i32 436, !"typeid2"}
!111 = !{i32 440, !"typeid2"}
!112 = !{i32 444, !"typeid2"}
!113 = !{i32 448, !"typeid2"}
!114 = !{i32 452, !"typeid2"}
!115 = !{i32 456, !"typeid2"}
!116 = !{i32 460, !"typeid2"}
!117 = !{i32 464, !"typeid2"}
!118 = !{i32 468, !"typeid2"}
!119 = !{i32 472, !"typeid2"}
!120 = !{i32 476, !"typeid2"}
!121 = !{i32 480, !"typeid2"}
!122 = !{i32 484, !"typeid2"}
!123 = !{i32 488, !"typeid2"}
!124 = !{i32 492, !"typeid2"}
!125 = !{i32 496, !"typeid2"}
!126 = !{i32 500, !"typeid2"}
!127 = !{i32 504, !"typeid2"}
!128 = !{i32 508, !"typeid2"}
!129 = !{i32 512, !"typeid2"}
!130 = !{i32 516, !"typeid2"}

; CHECK: [[G:@[0-9]+]] = private constant { [2048 x i8] } zeroinitializer

; CHECK: @__typeid_typeid1_global_addr = hidden alias i8, getelementptr inbounds ({ [2048 x i8] }, { [2048 x i8] }* [[G]], i32 0, i32 0, i32 0)
; X86: @__typeid_typeid1_align = hidden alias i8, inttoptr (i8 1 to i8*)
; X86: @__typeid_typeid1_size_m1 = hidden alias i8, inttoptr (i64 1 to i8*)

; CHECK: @__typeid_typeid2_global_addr = hidden alias i8, getelementptr inbounds ({ [2048 x i8] }, { [2048 x i8] }* [[G]], i32 0, i32 0, i64 4)
; X86: @__typeid_typeid2_align = hidden alias i8, inttoptr (i8 2 to i8*)
; X86: @__typeid_typeid2_size_m1 = hidden alias i8, inttoptr (i64 128 to i8*)

; ARM-NOT: alias {{.*}} inttoptr

; CHECK: @foo = alias [2048 x i8], getelementptr inbounds ({ [2048 x i8] }, { [2048 x i8] }* [[G]], i32 0, i32 0)

; SUMMARY:      TypeIdMap:
; SUMMARY-NEXT:   typeid1:
; SUMMARY-NEXT:     TTRes:
; SUMMARY-NEXT:       Kind:            AllOnes
; SUMMARY-NEXT:       SizeM1BitWidth:  7
; SUMMARY-X86-NEXT:   AlignLog2:       0
; SUMMARY-X86-NEXT:   SizeM1:          0
; SUMMARY-X86-NEXT:   BitMask:         0
; SUMMARY-X86-NEXT:   InlineBits:      0
; SUMMARY-ARM-NEXT:   AlignLog2:       1
; SUMMARY-ARM-NEXT:   SizeM1:          1
; SUMMARY-ARM-NEXT:   BitMask:         0
; SUMMARY-ARM-NEXT:   InlineBits:      0
; SUMMARY-NEXT:     WPDRes:
; SUMMARY-NEXT:   typeid2:
; SUMMARY-NEXT:     TTRes:
; SUMMARY-NEXT:       Kind:            AllOnes
; SUMMARY-NEXT:       SizeM1BitWidth:  32
; SUMMARY-X86-NEXT:   AlignLog2:       0
; SUMMARY-X86-NEXT:   SizeM1:          0
; SUMMARY-X86-NEXT:   BitMask:         0
; SUMMARY-X86-NEXT:   InlineBits:      0
; SUMMARY-ARM-NEXT:   AlignLog2:       2
; SUMMARY-ARM-NEXT:   SizeM1:          128
; SUMMARY-ARM-NEXT:   BitMask:         0
; SUMMARY-ARM-NEXT:   InlineBits:      0
; SUMMARY-NEXT:     WPDRes:
