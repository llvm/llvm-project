; RUN: opt -S -passes=gvn-hoist < %s | FileCheck %s

; Check that the stores are not hoisted: it is invalid to hoist stores if they
; are not executed on all paths. In this testcase, there are paths in the loop
; that do not execute the stores.

; CHECK-LABEL: define void @music_task
; CHECK: store
; CHECK: store
; CHECK: store


%struct._MUSIC_OP_API_ = type { ptr, ptr }
%struct._FILE_OPERATE_ = type { ptr, ptr }
%struct._FILE_OPERATE_INIT_ = type { i32, i32, i32, i32, ptr, ptr, i32 }
%struct._lg_dev_info_ = type { %struct.os_event, i32, i32, ptr, i8, i8, i8, i8, i8 }
%struct.os_event = type { i8, i32, ptr, %union.anon }
%union.anon = type { %struct.event_cnt }
%struct.event_cnt = type { i16 }
%struct._lg_dev_hdl_ = type { ptr, ptr, ptr, ptr, ptr }
%struct.__MUSIC_API = type <{ ptr, ptr, i32, %struct._DEC_API, ptr, ptr }>
%struct._DEC_API = type { ptr, ptr, ptr, ptr, ptr, ptr, %struct._AAC_DEFAULT_SETTING, i32, i32, ptr, ptr, i32, i8, ptr, i8, ptr }
%struct._DEC_PHY = type { ptr, ptr, ptr, %struct.if_decoder_io, ptr, ptr, ptr, i32, i8, %struct.__FF_FR }
%struct.__audio_decoder_ops = type { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }
%struct.if_decoder_io = type { ptr, ptr, ptr, ptr, ptr, ptr }
%struct.if_dec_file = type { ptr, ptr }
%struct.__FF_FR = type { i32, i32, i8, i8, i8 }
%struct._AAC_DEFAULT_SETTING = type { i32, i32, i32 }
%struct.decoder_inf = type { i16, i16, i32, i32 }
%struct._DEC_API_IO = type { ptr, ptr, ptr, ptr, ptr, %struct.__OP_IO, i32, i32 }
%struct.__OP_IO = type { ptr, ptr }
%struct._FS_BRK_POINT = type { %struct._FS_BRK_INFO, i32, i32 }
%struct._FS_BRK_INFO = type { i32, i32, [8 x i8], i8, i8, i16 }

@.str = external hidden unnamed_addr constant [10 x i8], align 1

define void @music_task(ptr nocapture readnone %p) local_unnamed_addr {
entry:
  %mapi = alloca ptr, align 8
  call void @llvm.lifetime.start.p0(i64 8, ptr %mapi)
  store ptr null, ptr %mapi, align 8, !tbaa !1
  %call = call i32 @music_decoder_init(ptr nonnull %mapi)
  br label %while.cond

while.cond.loopexit:                              ; preds = %while.cond2
  br label %while.cond

while.cond:                                       ; preds = %while.cond.loopexit, %entry
  %0 = load ptr, ptr %mapi, align 8, !tbaa !1
  %dop_api = getelementptr inbounds %struct._MUSIC_OP_API_, ptr %0, i64 0, i32 1
  %1 = load ptr, ptr %dop_api, align 8, !tbaa !5
  %file_num = getelementptr inbounds %struct.__MUSIC_API, ptr %1, i64 0, i32 2
  %call1 = call i32 @music_play_api(ptr %0, i32 33, i32 0, i32 28, ptr %file_num)
  br label %while.cond2

while.cond2:                                      ; preds = %while.cond2.backedge, %while.cond
  %err.0 = phi i32 [ %call1, %while.cond ], [ %err.0.be, %while.cond2.backedge ]
  switch i32 %err.0, label %sw.default [
    i32 0, label %while.cond.loopexit
    i32 35, label %sw.bb
    i32 11, label %sw.bb7
    i32 12, label %sw.bb13
  ]

sw.bb:                                            ; preds = %while.cond2
  %2 = load ptr, ptr %mapi, align 8, !tbaa !1
  %dop_api4 = getelementptr inbounds %struct._MUSIC_OP_API_, ptr %2, i64 0, i32 1
  %3 = load ptr, ptr %dop_api4, align 8, !tbaa !5
  %file_num5 = getelementptr inbounds %struct.__MUSIC_API, ptr %3, i64 0, i32 2
  %4 = load i32, ptr %file_num5, align 1, !tbaa !7
  %call6 = call i32 (ptr, ...) @printf(ptr @.str, i32 %4)
  br label %while.cond2.backedge

sw.bb7:                                           ; preds = %while.cond2
  %5 = load ptr, ptr %mapi, align 8, !tbaa !1
  %dop_api8 = getelementptr inbounds %struct._MUSIC_OP_API_, ptr %5, i64 0, i32 1
  %6 = load ptr, ptr %dop_api8, align 8, !tbaa !5
  %file_num9 = getelementptr inbounds %struct.__MUSIC_API, ptr %6, i64 0, i32 2
  store i32 1, ptr %file_num9, align 1, !tbaa !7
  %call12 = call i32 @music_play_api(ptr %5, i32 34, i32 0, i32 24, ptr %file_num9)
  br label %while.cond2.backedge

sw.bb13:                                          ; preds = %while.cond2
  %7 = load ptr, ptr %mapi, align 8, !tbaa !1
  %dop_api14 = getelementptr inbounds %struct._MUSIC_OP_API_, ptr %7, i64 0, i32 1
  %8 = load ptr, ptr %dop_api14, align 8, !tbaa !5
  %file_num15 = getelementptr inbounds %struct.__MUSIC_API, ptr %8, i64 0, i32 2
  store i32 1, ptr %file_num15, align 1, !tbaa !7
  %call18 = call i32 @music_play_api(ptr %7, i32 35, i32 0, i32 26, ptr %file_num15)
  br label %while.cond2.backedge

sw.default:                                       ; preds = %while.cond2
  %9 = load ptr, ptr %mapi, align 8, !tbaa !1
  %call19 = call i32 @music_play_api(ptr %9, i32 33, i32 0, i32 22, ptr null)
  br label %while.cond2.backedge

while.cond2.backedge:                             ; preds = %sw.default, %sw.bb13, %sw.bb7, %sw.bb
  %err.0.be = phi i32 [ %call19, %sw.default ], [ %call18, %sw.bb13 ], [ %call12, %sw.bb7 ], [ 0, %sw.bb ]
  br label %while.cond2
}

declare void @llvm.lifetime.start.p0(i64, ptr nocapture)
declare i32 @music_decoder_init(ptr)
declare i32 @music_play_api(ptr, i32, i32, i32, ptr)
declare i32 @printf(ptr nocapture readonly, ...)

!0 = !{!"clang version 4.0.0 "}
!1 = !{!2, !2, i64 0}
!2 = !{!"any pointer", !3, i64 0}
!3 = !{!"omnipotent char", !4, i64 0}
!4 = !{!"Simple C/C++ TBAA"}
!5 = !{!6, !2, i64 8}
!6 = !{!"_MUSIC_OP_API_", !2, i64 0, !2, i64 8}
!7 = !{!8, !9, i64 16}
!8 = !{!"__MUSIC_API", !2, i64 0, !2, i64 8, !9, i64 16, !10, i64 20, !2, i64 140, !2, i64 148}
!9 = !{!"int", !3, i64 0}
!10 = !{!"_DEC_API", !2, i64 0, !2, i64 8, !2, i64 16, !2, i64 24, !2, i64 32, !2, i64 40, !11, i64 48, !9, i64 60, !9, i64 64, !2, i64 72, !2, i64 80, !9, i64 88, !3, i64 92, !2, i64 96, !3, i64 104, !2, i64 112}
!11 = !{!"_AAC_DEFAULT_SETTING", !9, i64 0, !9, i64 4, !9, i64 8}
