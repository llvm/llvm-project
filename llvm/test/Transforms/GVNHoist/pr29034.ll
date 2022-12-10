; RUN: opt -S -passes=gvn-hoist < %s | FileCheck %s

; Check that the stores are not hoisted: it is invalid to hoist stores if they
; are not executed on all paths. In this testcase, there are paths in the loop
; that do not execute the stores.

; CHECK-LABEL: define void @music_task
; CHECK: store
; CHECK: store
; CHECK: store


%struct._MUSIC_OP_API_ = type { %struct._FILE_OPERATE_*, %struct.__MUSIC_API* }
%struct._FILE_OPERATE_ = type { %struct._FILE_OPERATE_INIT_*, %struct._lg_dev_info_* }
%struct._FILE_OPERATE_INIT_ = type { i32, i32, i32, i32, i32*, i8*, i32 }
%struct._lg_dev_info_ = type { %struct.os_event, i32, i32, %struct._lg_dev_hdl_*, i8, i8, i8, i8, i8 }
%struct.os_event = type { i8, i32, i8*, %union.anon }
%union.anon = type { %struct.event_cnt }
%struct.event_cnt = type { i16 }
%struct._lg_dev_hdl_ = type { i8*, i8*, i8*, i8*, i8* }
%struct.__MUSIC_API = type <{ i8*, i8*, i32, %struct._DEC_API, %struct._DEC_API_IO*, %struct._FS_BRK_POINT* }>
%struct._DEC_API = type { %struct._DEC_PHY*, i8*, i8*, i8* (i8*)*, i32* (i8*)*, i8*, %struct._AAC_DEFAULT_SETTING, i32, i32, i8*, %struct.decoder_inf*, i32, i8, i8*, i8, i8* }
%struct._DEC_PHY = type { i8*, %struct.__audio_decoder_ops*, i8*, %struct.if_decoder_io, %struct.if_dec_file*, i8*, i32 (i8*)*, i32, i8, %struct.__FF_FR }
%struct.__audio_decoder_ops = type { i8*, i32 (i8*, %struct.if_decoder_io*, i8*)*, i32 (i8*)*, i32 (i8*, i32)*, %struct.decoder_inf* (i8*)*, i32 (i8*)*, i32 (i8*)*, i32 (...)*, i32 (...)*, i32 (...)*, void (i8*, i32)*, void (i8*, i32, i8*, i32)*, i32 (i8*, i32, i8*)* }
%struct.if_decoder_io = type { i8*, i32 (i8*, i32, i8*, i32, i8)*, i32 (i8*, i32, i8*)*, void (i8*, i8*, i32)*, i32 (i8*)*, i32 (i8*, i32, i32)* }
%struct.if_dec_file = type { i32 (i8*, i8*, i32)*, i32 (i8*, i32, i32)* }
%struct.__FF_FR = type { i32, i32, i8, i8, i8 }
%struct._AAC_DEFAULT_SETTING = type { i32, i32, i32 }
%struct.decoder_inf = type { i16, i16, i32, i32 }
%struct._DEC_API_IO = type { i8*, i8*, i16 (i8*, i8*, i16)*, i32 (i8*, i8, i32)*, i32 (%struct.decoder_inf*, i32)*, %struct.__OP_IO, i32, i32 }
%struct.__OP_IO = type { i8*, i8* (i8*, i8*, i32)* }
%struct._FS_BRK_POINT = type { %struct._FS_BRK_INFO, i32, i32 }
%struct._FS_BRK_INFO = type { i32, i32, [8 x i8], i8, i8, i16 }

@.str = external hidden unnamed_addr constant [10 x i8], align 1

define void @music_task(i8* nocapture readnone %p) local_unnamed_addr {
entry:
  %mapi = alloca %struct._MUSIC_OP_API_*, align 8
  %0 = bitcast %struct._MUSIC_OP_API_** %mapi to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* %0)
  store %struct._MUSIC_OP_API_* null, %struct._MUSIC_OP_API_** %mapi, align 8, !tbaa !1
  %call = call i32 @music_decoder_init(%struct._MUSIC_OP_API_** nonnull %mapi)
  br label %while.cond

while.cond.loopexit:                              ; preds = %while.cond2
  br label %while.cond

while.cond:                                       ; preds = %while.cond.loopexit, %entry
  %1 = load %struct._MUSIC_OP_API_*, %struct._MUSIC_OP_API_** %mapi, align 8, !tbaa !1
  %dop_api = getelementptr inbounds %struct._MUSIC_OP_API_, %struct._MUSIC_OP_API_* %1, i64 0, i32 1
  %2 = load %struct.__MUSIC_API*, %struct.__MUSIC_API** %dop_api, align 8, !tbaa !5
  %file_num = getelementptr inbounds %struct.__MUSIC_API, %struct.__MUSIC_API* %2, i64 0, i32 2
  %3 = bitcast i32* %file_num to i8*
  %call1 = call i32 @music_play_api(%struct._MUSIC_OP_API_* %1, i32 33, i32 0, i32 28, i8* %3)
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
  %4 = load %struct._MUSIC_OP_API_*, %struct._MUSIC_OP_API_** %mapi, align 8, !tbaa !1
  %dop_api4 = getelementptr inbounds %struct._MUSIC_OP_API_, %struct._MUSIC_OP_API_* %4, i64 0, i32 1
  %5 = load %struct.__MUSIC_API*, %struct.__MUSIC_API** %dop_api4, align 8, !tbaa !5
  %file_num5 = getelementptr inbounds %struct.__MUSIC_API, %struct.__MUSIC_API* %5, i64 0, i32 2
  %6 = load i32, i32* %file_num5, align 1, !tbaa !7
  %call6 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([10 x i8], [10 x i8]* @.str, i64 0, i64 0), i32 %6)
  br label %while.cond2.backedge

sw.bb7:                                           ; preds = %while.cond2
  %7 = load %struct._MUSIC_OP_API_*, %struct._MUSIC_OP_API_** %mapi, align 8, !tbaa !1
  %dop_api8 = getelementptr inbounds %struct._MUSIC_OP_API_, %struct._MUSIC_OP_API_* %7, i64 0, i32 1
  %8 = load %struct.__MUSIC_API*, %struct.__MUSIC_API** %dop_api8, align 8, !tbaa !5
  %file_num9 = getelementptr inbounds %struct.__MUSIC_API, %struct.__MUSIC_API* %8, i64 0, i32 2
  store i32 1, i32* %file_num9, align 1, !tbaa !7
  %9 = bitcast i32* %file_num9 to i8*
  %call12 = call i32 @music_play_api(%struct._MUSIC_OP_API_* %7, i32 34, i32 0, i32 24, i8* %9)
  br label %while.cond2.backedge

sw.bb13:                                          ; preds = %while.cond2
  %10 = load %struct._MUSIC_OP_API_*, %struct._MUSIC_OP_API_** %mapi, align 8, !tbaa !1
  %dop_api14 = getelementptr inbounds %struct._MUSIC_OP_API_, %struct._MUSIC_OP_API_* %10, i64 0, i32 1
  %11 = load %struct.__MUSIC_API*, %struct.__MUSIC_API** %dop_api14, align 8, !tbaa !5
  %file_num15 = getelementptr inbounds %struct.__MUSIC_API, %struct.__MUSIC_API* %11, i64 0, i32 2
  store i32 1, i32* %file_num15, align 1, !tbaa !7
  %12 = bitcast i32* %file_num15 to i8*
  %call18 = call i32 @music_play_api(%struct._MUSIC_OP_API_* %10, i32 35, i32 0, i32 26, i8* %12)
  br label %while.cond2.backedge

sw.default:                                       ; preds = %while.cond2
  %13 = load %struct._MUSIC_OP_API_*, %struct._MUSIC_OP_API_** %mapi, align 8, !tbaa !1
  %call19 = call i32 @music_play_api(%struct._MUSIC_OP_API_* %13, i32 33, i32 0, i32 22, i8* null)
  br label %while.cond2.backedge

while.cond2.backedge:                             ; preds = %sw.default, %sw.bb13, %sw.bb7, %sw.bb
  %err.0.be = phi i32 [ %call19, %sw.default ], [ %call18, %sw.bb13 ], [ %call12, %sw.bb7 ], [ 0, %sw.bb ]
  br label %while.cond2
}

declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture)
declare i32 @music_decoder_init(%struct._MUSIC_OP_API_**)
declare i32 @music_play_api(%struct._MUSIC_OP_API_*, i32, i32, i32, i8*)
declare i32 @printf(i8* nocapture readonly, ...)

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
