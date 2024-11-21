;RUN: llc  -stop-after="hexagon-isel"  < %s | FileCheck %s

;CHECK: %54:intregs = L2_loadrub_io %27, -1
target triple = "hexagon"

@global = external dso_local local_unnamed_addr global ptr, align 4

; Function Attrs: nofree norecurse nosync nounwind memory(read, argmem: readwrite, inaccessiblemem: none)
define dso_local void @quux(i32 noundef %arg, ptr nocapture noundef readonly %arg1, ptr nocapture noundef writeonly %arg2, i32 noundef %arg3) local_unnamed_addr #0 {
bb:
  %getelementptr = getelementptr inbounds nuw i8, ptr %arg1, i32 29
  %load = load i8, ptr %getelementptr, align 1, !tbaa !3
  %icmp = icmp eq i8 %load, 3
  br i1 %icmp, label %bb4, label %bb170

bb4:                                              ; preds = %bb
  %load5 = load ptr, ptr @global, align 4, !tbaa !9
  %getelementptr6 = getelementptr inbounds nuw i8, ptr %arg1, i32 4
  %load7 = load ptr, ptr %getelementptr6, align 4, !tbaa !10
  %getelementptr8 = getelementptr inbounds nuw i8, ptr %load7, i32 8
  %load9 = load i32, ptr %getelementptr8, align 4, !tbaa !11
  %getelementptr10 = getelementptr inbounds nuw i8, ptr %load5, i32 %load9
  %getelementptr11 = getelementptr inbounds i8, ptr %getelementptr10, i32 -1
  %sub = sub i32 0, %arg3
  %getelementptr12 = getelementptr inbounds i8, ptr %getelementptr10, i32 %sub
  %getelementptr13 = getelementptr inbounds nuw i8, ptr %getelementptr12, i32 4
  %load14 = load i8, ptr %getelementptr13, align 1, !tbaa !14
  %zext = zext i8 %load14 to i32
  %getelementptr15 = getelementptr inbounds nuw i8, ptr %getelementptr12, i32 2
  %load16 = load i8, ptr %getelementptr15, align 1, !tbaa !14
  %zext17 = zext i8 %load16 to i32
  %sub18 = sub nsw i32 %zext, %zext17
  %shl = shl i32 %arg3, 2
  %getelementptr19 = getelementptr inbounds nuw i8, ptr %getelementptr11, i32 %shl
  %load20 = load i8, ptr %getelementptr19, align 1, !tbaa !14
  %zext21 = zext i8 %load20 to i32
  %shl22 = shl i32 %arg3, 1
  %getelementptr23 = getelementptr inbounds nuw i8, ptr %getelementptr11, i32 %shl22
  %load24 = load i8, ptr %getelementptr23, align 1, !tbaa !14
  %zext25 = zext i8 %load24 to i32
  %sub26 = sub nsw i32 %zext21, %zext25
  %getelementptr27 = getelementptr inbounds nuw i8, ptr %getelementptr12, i32 5
  %load28 = load i8, ptr %getelementptr27, align 1, !tbaa !14
  %zext29 = zext i8 %load28 to i32
  %getelementptr30 = getelementptr inbounds nuw i8, ptr %getelementptr12, i32 1
  %load31 = load i8, ptr %getelementptr30, align 1, !tbaa !14
  %zext32 = zext i8 %load31 to i32
  %sub33 = sub nsw i32 %zext29, %zext32
  %shl34 = shl nsw i32 %sub33, 1
  %add = add nsw i32 %shl34, %sub18
  %mul = mul i32 %arg3, 5
  %getelementptr35 = getelementptr inbounds nuw i8, ptr %getelementptr11, i32 %mul
  %load36 = load i8, ptr %getelementptr35, align 1, !tbaa !14
  %zext37 = zext i8 %load36 to i32
  %getelementptr38 = getelementptr inbounds nuw i8, ptr %getelementptr11, i32 %arg3
  %load39 = load i8, ptr %getelementptr38, align 1, !tbaa !14
  %zext40 = zext i8 %load39 to i32
  %sub41 = sub nsw i32 %zext37, %zext40
  %shl42 = shl nsw i32 %sub41, 1
  %add43 = add nsw i32 %shl42, %sub26
  %getelementptr44 = getelementptr inbounds nuw i8, ptr %getelementptr12, i32 6
  %load45 = load i8, ptr %getelementptr44, align 1, !tbaa !14
  %zext46 = zext i8 %load45 to i32
  %load47 = load i8, ptr %getelementptr12, align 1, !tbaa !14
  %zext48 = zext i8 %load47 to i32
  %sub49 = sub nsw i32 %zext46, %zext48
  %mul50 = mul nsw i32 %sub49, 3
  %add51 = add nsw i32 %mul50, %add
  %mul53 = mul i32 %arg3, 6
  %getelementptr54 = getelementptr inbounds nuw i8, ptr %getelementptr11, i32 %mul53
  %load55 = load i8, ptr %getelementptr54, align 1, !tbaa !14
  %zext56 = zext i8 %load55 to i32
  %load57 = load i8, ptr %getelementptr11, align 1, !tbaa !14
  %zext58 = zext i8 %load57 to i32
  %sub59 = sub nsw i32 %zext56, %zext58
  %mul60 = mul nsw i32 %sub59, 3
  %add61 = add nsw i32 %mul60, %add43
  %getelementptr62 = getelementptr inbounds i8, ptr %getelementptr12, i32 7
  %load63 = load i8, ptr %getelementptr62, align 1, !tbaa !14
  %zext64 = zext i8 %load63 to i32
  %getelementptr65 = getelementptr inbounds nuw i8, ptr %getelementptr12, i32 -1
  %load66 = load i8, ptr %getelementptr65, align 1, !tbaa !14
  %zext67 = zext i8 %load66 to i32
  %sub68 = sub nsw i32 %zext64, %zext67
  %shl69 = shl nsw i32 %sub68, 2
  %add70 = add nsw i32 %shl69, %add51
  %mul71 = mul i32 %arg3, 7
  %getelementptr72 = getelementptr inbounds i8, ptr %getelementptr11, i32 %mul71
  %load73 = load i8, ptr %getelementptr72, align 1, !tbaa !14
  %zext74 = zext i8 %load73 to i32
  %sub75 = sub i32 0, %arg3
  %getelementptr76 = getelementptr inbounds nuw i8, ptr %getelementptr11, i32 %sub75
  %load77 = load i8, ptr %getelementptr76, align 1, !tbaa !14
  %zext78 = zext i8 %load77 to i32
  %sub79 = sub nsw i32 %zext74, %zext78
  %shl80 = shl nsw i32 %sub79, 2
  %add81 = add nsw i32 %shl80, %add61
  %add82 = add nuw nsw i32 %zext74, %zext64
  %shl83 = shl nuw nsw i32 %add82, 4
  %mul84 = mul nsw i32 %add70, 17
  %add85 = add nsw i32 %mul84, 16
  %ashr = ashr i32 %add85, 5
  %mul86 = mul nsw i32 %add81, 17
  %add87 = add nsw i32 %mul86, 16
  %ashr88 = ashr i32 %add87, 5
  %add89 = add nuw nsw i32 %shl83, 16
  %mul90 = mul nsw i32 %ashr, -3
  %shl91 = shl nsw i32 %ashr, 1
  %mul92 = mul nsw i32 %ashr, 3
  %shl93 = shl nsw i32 %ashr, 2
  %shl94 = shl nsw i32 %ashr, 1
  br label %bb95

bb95:                                             ; preds = %bb95, %bb4
  %phi = phi ptr [ %arg2, %bb4 ], [ %getelementptr167, %bb95 ]
  %phi96 = phi i32 [ 0, %bb4 ], [ %add168, %bb95 ]
  %add97 = add nsw i32 %phi96, -3
  %mul98 = mul nsw i32 %add97, %ashr88
  %add99 = add i32 %add89, %mul98
  %add100 = add i32 %add99, %mul90
  %ashr101 = ashr i32 %add100, 5
  %and = and i32 %add100, 8192
  %icmp102 = icmp eq i32 %and, 0
  %icmp103 = icmp slt i32 %ashr101, 0
  %select = select i1 %icmp103, i32 0, i32 255
  %select104 = select i1 %icmp102, i32 %ashr101, i32 %select
  %trunc = trunc i32 %select104 to i8
  store i8 %trunc, ptr %phi, align 1, !tbaa !14
  %getelementptr105 = getelementptr inbounds nuw i8, ptr %phi, i32 1
  %sub106 = sub i32 %add99, %shl94
  %ashr107 = ashr i32 %sub106, 5
  %and108 = and i32 %sub106, 8192
  %icmp109 = icmp eq i32 %and108, 0
  %icmp110 = icmp slt i32 %ashr107, 0
  %select111 = select i1 %icmp110, i32 0, i32 255
  %select112 = select i1 %icmp109, i32 %ashr107, i32 %select111
  %trunc113 = trunc i32 %select112 to i8
  store i8 %trunc113, ptr %getelementptr105, align 1, !tbaa !14
  %getelementptr114 = getelementptr inbounds nuw i8, ptr %phi, i32 2
  %sub115 = sub i32 %add99, %ashr
  %ashr116 = ashr i32 %sub115, 5
  %and117 = and i32 %sub115, 8192
  %icmp118 = icmp eq i32 %and117, 0
  %icmp119 = icmp slt i32 %ashr116, 0
  %select120 = select i1 %icmp119, i32 0, i32 255
  %select121 = select i1 %icmp118, i32 %ashr116, i32 %select120
  %trunc122 = trunc i32 %select121 to i8
  store i8 %trunc122, ptr %getelementptr114, align 1, !tbaa !14
  %getelementptr123 = getelementptr inbounds nuw i8, ptr %phi, i32 3
  %ashr124 = ashr i32 %add99, 5
  %and125 = and i32 %add99, 8192
  %icmp126 = icmp eq i32 %and125, 0
  %icmp127 = icmp slt i32 %ashr124, 0
  %select128 = select i1 %icmp127, i32 0, i32 255
  %select129 = select i1 %icmp126, i32 %ashr124, i32 %select128
  %trunc130 = trunc i32 %select129 to i8
  store i8 %trunc130, ptr %getelementptr123, align 1, !tbaa !14
  %getelementptr131 = getelementptr inbounds nuw i8, ptr %phi, i32 4
  %add132 = add i32 %add99, %ashr
  %ashr133 = ashr i32 %add132, 5
  %and134 = and i32 %add132, 8192
  %icmp135 = icmp eq i32 %and134, 0
  %icmp136 = icmp slt i32 %ashr133, 0
  %select137 = select i1 %icmp136, i32 0, i32 255
  %select138 = select i1 %icmp135, i32 %ashr133, i32 %select137
  %trunc139 = trunc i32 %select138 to i8
  store i8 %trunc139, ptr %getelementptr131, align 1, !tbaa !14
  %getelementptr140 = getelementptr inbounds nuw i8, ptr %phi, i32 5
  %add141 = add i32 %add99, %shl91
  %ashr142 = ashr i32 %add141, 5
  %and143 = and i32 %add141, 8192
  %icmp144 = icmp eq i32 %and143, 0
  %icmp145 = icmp slt i32 %ashr142, 0
  %select146 = select i1 %icmp145, i32 0, i32 255
  %select147 = select i1 %icmp144, i32 %ashr142, i32 %select146
  %trunc148 = trunc i32 %select147 to i8
  store i8 %trunc148, ptr %getelementptr140, align 1, !tbaa !14
  %getelementptr149 = getelementptr inbounds nuw i8, ptr %phi, i32 6
  %add150 = add i32 %add99, %mul92
  %ashr151 = ashr i32 %add150, 5
  %and152 = and i32 %add150, 8192
  %icmp153 = icmp eq i32 %and152, 0
  %icmp154 = icmp slt i32 %ashr151, 0
  %select155 = select i1 %icmp154, i32 0, i32 255
  %select156 = select i1 %icmp153, i32 %ashr151, i32 %select155
  %trunc157 = trunc i32 %select156 to i8
  store i8 %trunc157, ptr %getelementptr149, align 1, !tbaa !14
  %getelementptr158 = getelementptr inbounds nuw i8, ptr %phi, i32 7
  %add159 = add i32 %add99, %shl93
  %ashr160 = ashr i32 %add159, 5
  %and161 = and i32 %add159, 8192
  %icmp162 = icmp eq i32 %and161, 0
  %icmp163 = icmp slt i32 %ashr160, 0
  %select164 = select i1 %icmp163, i32 0, i32 255
  %select165 = select i1 %icmp162, i32 %ashr160, i32 %select164
  %trunc166 = trunc i32 %select165 to i8
  store i8 %trunc166, ptr %getelementptr158, align 1, !tbaa !14
  %getelementptr167 = getelementptr inbounds nuw i8, ptr %phi, i32 8
  %add168 = add nuw nsw i32 %phi96, 1
  %icmp169 = icmp eq i32 %add168, 8
  br i1 %icmp169, label %bb170, label %bb95, !llvm.loop !15

bb170:                                            ; preds = %bb95, %bb
  ret void
}

attributes #0 = { nofree norecurse nosync nounwind memory(read, argmem: readwrite, inaccessiblemem: none) "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="hexagonv68" "target-features"="+v68,-long-calls" }

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"frame-pointer", i32 2}
!2 = !{!"QuIC LLVM Hexagon Clang version 8.9 Engineering Release: hexagon-clang-89"}
!3 = !{!4, !6, i64 29}
!4 = !{!"MB_STRUCT", !5, i64 0, !5, i64 4, !5, i64 8, !8, i64 12, !8, i64 16, !8, i64 20, !8, i64 24, !6, i64 28, !6, i64 29, !5, i64 32}
!5 = !{!"any pointer", !6, i64 0}
!6 = !{!"omnipotent char", !7, i64 0}
!7 = !{!"Simple C/C++ TBAA"}
!8 = !{!"int", !6, i64 0}
!9 = !{!5, !5, i64 0}
!10 = !{!4, !5, i64 4}
!11 = !{!12, !8, i64 8}
!12 = !{!"BLOCK_STRUCT_CHROM", !5, i64 0, !5, i64 4, !8, i64 8, !13, i64 12, !13, i64 14, !6, i64 16, !6, i64 17, !6, i64 18, !6, i64 19}
!13 = !{!"short", !6, i64 0}
!14 = !{!6, !6, i64 0}
!15 = distinct !{!15, !16}
!16 = !{!"llvm.loop.mustprogress"}
