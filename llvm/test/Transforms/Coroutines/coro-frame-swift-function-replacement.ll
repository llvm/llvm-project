; RUN: opt < %s -coro-split -S | FileCheck %s

; Make sure we can handle dynamically replaceable swift functions

target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.15.0"

%swift.dyn_repl_link_entry = type { i8*, %swift.dyn_repl_link_entry* }
%swift.type = type { i64 }
%swift.bridge = type opaque
%TSi = type <{ i64 }>
%T13CoroFrameRepo9ContainerV = type <{ %TSa }>
%TSa = type <{ %Ts12_ArrayBufferV }>
%Ts12_ArrayBufferV = type <{ %Ts14_BridgeStorageV }>
%Ts14_BridgeStorageV = type <{ %swift.bridge* }>
%Any = type { [24 x i8], %swift.type* }

@"$s13CoroFrameRepo9ContainerVyS2iciMTX" = external global %swift.dyn_repl_link_entry, align 8
@"$sSiN" = external global %swift.type, align 8
declare i8* @malloc(i64)
declare void @free(i8*)

declare token @llvm.coro.id.retcon.once(i32, i32, i8*, i8*, i8*, i8*) #0

declare i8* @llvm.coro.begin(token, i8* writeonly) #0

declare i1 @llvm.coro.suspend.retcon.i1(...) #0

declare i1 @llvm.coro.end(i8*, i1) #0

declare i8* @swift_getFunctionReplacement(i8**, i8*) #0
declare i8* @swift_getFunctionReplacement50(i8**, i8*) #0

declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i1 immarg) #1

declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #2

declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture) #2

; CHECK-LABEL: define swiftcc { i8*, %TSi* } @"$s13CoroFrameRepo9ContainerVyS2iciM"(i8* noalias dereferenceable(32) %arg, i64 %arg1, %T13CoroFrameRepo9ContainerV* swiftself dereferenceable(8) %arg2)
; CHECK: entry:
; CHECK:   [[TMP:%.*]] = call i8* @swift_getFunctionReplacement
; CHECK:   [[TMP2:%.*]] = icmp eq i8* [[TMP]], null
; CHECK:   br i1 [[TMP2]], label %original_entry, label %forward_to_replaced

; CHECK: forward_to_replaced:
; CHECK:   [[TMP3:%.*]] = bitcast i8* [[TMP]]
; CHECK:   [[TMP4:%.*]] = tail call swiftcc { i8*, %TSi* } [[TMP3]]({{.*}} %arg, i64 %arg1, {{.*}} %arg2)
; CHECK:   ret { i8*, %TSi* } [[TMP4]]

; CHECK: original_entry:
; CHECK:   call i8* @malloc(

define swiftcc { i8*, %TSi* } @"$s13CoroFrameRepo9ContainerVyS2iciM"(i8* noalias dereferenceable(32) %arg, i64 %arg1, %T13CoroFrameRepo9ContainerV* nocapture swiftself dereferenceable(8) %arg2) #3 {
entry:
  %idx.debug = alloca i64, align 8
  %tmp = bitcast i64* %idx.debug to i8*
  call void @llvm.memset.p0i8.i64(i8* align 8 %tmp, i8 0, i64 8, i1 false)
  %tmp3 = alloca [32 x i8], align 8
  %tmp4 = call i8* @swift_getFunctionReplacement(i8** getelementptr inbounds (%swift.dyn_repl_link_entry, %swift.dyn_repl_link_entry* @"$s13CoroFrameRepo9ContainerVyS2iciMTX", i32 0, i32 0), i8* bitcast ({ i8*, %TSi* } (i8*, i64, %T13CoroFrameRepo9ContainerV*)* @"$s13CoroFrameRepo9ContainerVyS2iciM" to i8*)) #0
  %tmp5 = icmp eq i8* %tmp4, null
  br i1 %tmp5, label %original_entry, label %forward_to_replaced

forward_to_replaced:                              ; preds = %entry
  %tmp6 = bitcast i8* %tmp4 to { i8*, %TSi* } (i8*, i64, %T13CoroFrameRepo9ContainerV*)*
  %tmp7 = tail call swiftcc { i8*, %TSi* } %tmp6(i8* noalias dereferenceable(32) %arg, i64 %arg1, %T13CoroFrameRepo9ContainerV* nocapture swiftself dereferenceable(8) %arg2)
  ret { i8*, %TSi* } %tmp7

original_entry:                                   ; preds = %entry
  %tmp8 = call token @llvm.coro.id.retcon.once(i32 32, i32 8, i8* %arg, i8* bitcast (void (i8*, i1)* @"$sSi13CoroFrameRepo9ContainerVSiIetMAylYl_TC" to i8*), i8* bitcast (i8* (i64)* @malloc to i8*), i8* bitcast (void (i8*)* @free to i8*))
  %tmp9 = call i8* @llvm.coro.begin(token %tmp8, i8* null)
  store i64 %arg1, i64* %idx.debug, align 8
  %.data = getelementptr inbounds %T13CoroFrameRepo9ContainerV, %T13CoroFrameRepo9ContainerV* %arg2, i32 0, i32 0
  %tmp10 = getelementptr inbounds [32 x i8], [32 x i8]* %tmp3, i32 0, i32 0
  call void @llvm.lifetime.start.p0i8(i64 32, i8* %tmp10)
  %tmp11 = call i8* @llvm.coro.prepare.retcon(i8* bitcast ({ i8*, %TSi* } (i8*, i64, %TSa*)* @"$sSayxSiciMSi_Tg5" to i8*))
  %tmp12 = bitcast i8* %tmp11 to { i8*, %TSi* } (i8*, i64, %TSa*)*
  %tmp13 = call swiftcc { i8*, %TSi* } %tmp12(i8* noalias dereferenceable(32) %tmp10, i64 %arg1, %TSa* nocapture swiftself dereferenceable(8) %.data)
  %tmp14 = extractvalue { i8*, %TSi* } %tmp13, 0
  %tmp15 = extractvalue { i8*, %TSi* } %tmp13, 1
  %tmp16 = call i1 (...) @llvm.coro.suspend.retcon.i1(%TSi* %tmp15)
  br i1 %tmp16, label %bb35, label %bb

bb:                                               ; preds = %original_entry
  call void @llvm.lifetime.end.p0i8(i64 32, i8* %tmp10)
  %.data1 = getelementptr inbounds %T13CoroFrameRepo9ContainerV, %T13CoroFrameRepo9ContainerV* %arg2, i32 0, i32 0
  br label %coro.end

bb35:                                             ; preds = %original_entry
  %tmp36 = bitcast i8* %tmp14 to void (i8*, i1)*
  call swiftcc void %tmp36(i8* noalias dereferenceable(32) %tmp10, i1 false)
  call void @llvm.lifetime.end.p0i8(i64 32, i8* %tmp10)
  br label %coro.end

coro.end:                                         ; preds = %bb35, %bb
  %tmp37 = call i1 @llvm.coro.end(i8* %tmp9, i1 false) #6
  unreachable
}

; CHECK-LABEL: define swiftcc { i8*, %TSi* } @"$s13CoroFrameRepo9ContainerVyS2iciM2"(i8* noalias dereferenceable(32) %arg, i64 %arg1, %T13CoroFrameRepo9ContainerV* swiftself dereferenceable(8) %arg2)
; CHECK: entry:
; CHECK:   [[TMP:%.*]] = call i8* @swift_getFunctionReplacement50
; CHECK:   [[TMP2:%.*]] = icmp eq i8* [[TMP]], null
; CHECK:   br i1 [[TMP2]], label %original_entry, label %forward_to_replaced

; CHECK: forward_to_replaced:
; CHECK:   [[TMP3:%.*]] = bitcast i8* [[TMP]]
; CHECK:   [[TMP4:%.*]] = tail call swiftcc { i8*, %TSi* } [[TMP3]]({{.*}} %arg, i64 %arg1, {{.*}} %arg2)
; CHECK:   ret { i8*, %TSi* } [[TMP4]]

; CHECK: original_entry:
; CHECK:   call i8* @malloc(

define swiftcc { i8*, %TSi* } @"$s13CoroFrameRepo9ContainerVyS2iciM2"(i8* noalias dereferenceable(32) %arg, i64 %arg1, %T13CoroFrameRepo9ContainerV* nocapture swiftself dereferenceable(8) %arg2) #3 {
entry:
  %idx.debug = alloca i64, align 8
  %tmp = bitcast i64* %idx.debug to i8*
  call void @llvm.memset.p0i8.i64(i8* align 8 %tmp, i8 0, i64 8, i1 false)
  %tmp3 = alloca [32 x i8], align 8
  %tmp4 = call i8* @swift_getFunctionReplacement50(i8** getelementptr inbounds (%swift.dyn_repl_link_entry, %swift.dyn_repl_link_entry* @"$s13CoroFrameRepo9ContainerVyS2iciMTX", i32 0, i32 0), i8* bitcast ({ i8*, %TSi* } (i8*, i64, %T13CoroFrameRepo9ContainerV*)* @"$s13CoroFrameRepo9ContainerVyS2iciM" to i8*)) #0
  %tmp5 = icmp eq i8* %tmp4, null
  br i1 %tmp5, label %original_entry, label %forward_to_replaced

forward_to_replaced:                              ; preds = %entry
  %tmp6 = bitcast i8* %tmp4 to { i8*, %TSi* } (i8*, i64, %T13CoroFrameRepo9ContainerV*)*
  %tmp7 = tail call swiftcc { i8*, %TSi* } %tmp6(i8* noalias dereferenceable(32) %arg, i64 %arg1, %T13CoroFrameRepo9ContainerV* nocapture swiftself dereferenceable(8) %arg2)
  ret { i8*, %TSi* } %tmp7

original_entry:                                   ; preds = %entry
  %tmp8 = call token @llvm.coro.id.retcon.once(i32 32, i32 8, i8* %arg, i8* bitcast (void (i8*, i1)* @"$sSi13CoroFrameRepo9ContainerVSiIetMAylYl_TC" to i8*), i8* bitcast (i8* (i64)* @malloc to i8*), i8* bitcast (void (i8*)* @free to i8*))
  %tmp9 = call i8* @llvm.coro.begin(token %tmp8, i8* null)
  store i64 %arg1, i64* %idx.debug, align 8
  %.data = getelementptr inbounds %T13CoroFrameRepo9ContainerV, %T13CoroFrameRepo9ContainerV* %arg2, i32 0, i32 0
  %tmp10 = getelementptr inbounds [32 x i8], [32 x i8]* %tmp3, i32 0, i32 0
  call void @llvm.lifetime.start.p0i8(i64 32, i8* %tmp10)
  %tmp11 = call i8* @llvm.coro.prepare.retcon(i8* bitcast ({ i8*, %TSi* } (i8*, i64, %TSa*)* @"$sSayxSiciMSi_Tg5" to i8*))
  %tmp12 = bitcast i8* %tmp11 to { i8*, %TSi* } (i8*, i64, %TSa*)*
  %tmp13 = call swiftcc { i8*, %TSi* } %tmp12(i8* noalias dereferenceable(32) %tmp10, i64 %arg1, %TSa* nocapture swiftself dereferenceable(8) %.data)
  %tmp14 = extractvalue { i8*, %TSi* } %tmp13, 0
  %tmp15 = extractvalue { i8*, %TSi* } %tmp13, 1
  %tmp16 = call i1 (...) @llvm.coro.suspend.retcon.i1(%TSi* %tmp15)
  br i1 %tmp16, label %bb35, label %bb

bb:                                               ; preds = %original_entry
  call void @llvm.lifetime.end.p0i8(i64 32, i8* %tmp10)
  %.data1 = getelementptr inbounds %T13CoroFrameRepo9ContainerV, %T13CoroFrameRepo9ContainerV* %arg2, i32 0, i32 0
  br label %coro.end

bb35:                                             ; preds = %original_entry
  %tmp36 = bitcast i8* %tmp14 to void (i8*, i1)*
  call swiftcc void %tmp36(i8* noalias dereferenceable(32) %tmp10, i1 false)
  call void @llvm.lifetime.end.p0i8(i64 32, i8* %tmp10)
  br label %coro.end

coro.end:                                         ; preds = %bb35, %bb
  %tmp37 = call i1 @llvm.coro.end(i8* %tmp9, i1 false) #6
  unreachable
}
; Function Attrs: nounwind readnone
declare i8* @llvm.coro.prepare.retcon(i8*) #4
declare swiftcc void @"$sSi13CoroFrameRepo9ContainerVSiIetMAylYl_TC"(i8* noalias dereferenceable(32), i1) #5
declare swiftcc { i8*, %TSi* } @"$sSayxSiciMSi_Tg5"(i8* noalias dereferenceable(32), i64, %TSa* nocapture swiftself dereferenceable(8)) #5

attributes #0 = { nounwind }
attributes #1 = { argmemonly nounwind willreturn writeonly }
attributes #2 = { argmemonly nounwind willreturn }
attributes #3 = { noinline "coroutine.presplit"="1" "frame-pointer"="all" "probe-stack"="__chkstk_darwin" "target-cpu"="penryn" "target-features"="+cx16,+cx8,+fxsr,+mmx,+sahf,+sse,+sse2,+sse3,+sse4.1,+ssse3,+x87" }
attributes #4 = { nounwind readnone }
attributes #5 = { "frame-pointer"="all" "probe-stack"="__chkstk_darwin" "target-cpu"="penryn" "target-features"="+cx16,+cx8,+fxsr,+mmx,+sahf,+sse,+sse2,+sse3,+sse4.1,+ssse3,+x87" }
attributes #6 = { noduplicate }
