; RUN: opt -passes="cgscc(coro-annotation-elide)" -S < %s | FileCheck %s

%foo.Frame = type { ptr, ptr, i1 }

@foo.resumers = private constant [3 x ptr] [ptr @foo.resume, ptr @foo.destroy, ptr @foo.cleanup]
@foo.resumers.1 = private constant [4 x ptr] [ptr @foo.resume, ptr @foo.destroy, ptr @foo.cleanup, ptr @foo.noalloc]

; CHECK-LABEL: define void @foo
define void @foo(ptr %agg.result, ptr %this) personality ptr null {
entry:
  %0 = call token @llvm.coro.id(i32 0, ptr null, ptr nonnull @foo, ptr @foo.resumers.1)
  %1 = call noalias nonnull ptr @llvm.coro.begin(token %0, ptr null)
  %resume.addr = getelementptr inbounds nuw %foo.Frame, ptr %1, i32 0, i32 0
  store ptr @foo.resume, ptr %resume.addr, align 8
  %destroy.addr = getelementptr inbounds nuw %foo.Frame, ptr %1, i32 0, i32 1
  store ptr @foo.destroy, ptr %destroy.addr, align 8
  br label %AllocaSpillBB

AllocaSpillBB:                                    ; preds = %entry
  br label %PostSpill

PostSpill:                                        ; preds = %AllocaSpillBB
  br label %CoroSave

CoroSave:                                         ; preds = %PostSpill
  %index.addr1 = getelementptr inbounds nuw %foo.Frame, ptr %1, i32 0, i32 2
  store i1 false, ptr %index.addr1, align 1
  br label %CoroSuspend

CoroSuspend:                                      ; preds = %CoroSave
  br label %resume.0.landing

resume.0.landing:                                 ; preds = %CoroSuspend
  br label %AfterCoroSuspend

AfterCoroSuspend:                                 ; preds = %resume.0.landing
  ret void
}

; CHECK-LABEL: define internal void @bar
; Function Attrs: presplitcoroutine
define internal void @bar() #0 personality ptr null {
entry:
  ; CHECK: %[[CALLEE_FRAME:.+]] = alloca [24 x i8], align 8
  %0 = call token @llvm.coro.id(i32 0, ptr null, ptr nonnull @bar, ptr null)
  %1 = call i1 @llvm.coro.alloc(token %0)
  call void @foo(ptr null, ptr null) #4
  ; CHECK: %[[FOO_ID:.+]] = call token @llvm.coro.id(i32 0, ptr null, ptr nonnull @foo, ptr @foo.resumers)
  ; CHECK-NEXT: store ptr @foo.resume, ptr %[[CALLEE_FRAME]], align 8
  ; CHECK-NEXT: %[[DESTROY_ADDR:.+]] = getelementptr inbounds nuw %foo.Frame, ptr %[[CALLEE_FRAME]], i32 0, i32 1
  ; CHECK-NEXT: store ptr @foo.destroy, ptr %[[DESTROY_ADDR]], align 8
  ; CHECK-NEXT: %[[INDEX_ADDR:.+]] = getelementptr inbounds nuw %foo.Frame, ptr %[[CALLEE_FRAME]], i32 0, i32 2
  ; CHECK-NEXT: store i1 false, ptr %[[INDEX_ADDR]], align 1
  ; CHECK: ret void
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: read)
declare token @llvm.coro.id(i32, ptr readnone, ptr nocapture readonly, ptr) #1

; Function Attrs: nounwind
declare i1 @llvm.coro.alloc(token) #2

; Function Attrs: nounwind
declare ptr @llvm.coro.begin(token, ptr writeonly) #2

; Function Attrs: nomerge nounwind
declare token @llvm.coro.save(ptr) #3

; Function Attrs: nounwind
declare i8 @llvm.coro.suspend(token, i1) #2

define internal fastcc void @foo.resume(ptr noundef nonnull align 8 dereferenceable(24) %0) personality ptr null {
entry.resume:
  br label %resume.entry

resume.0:                                         ; preds = %resume.entry
  br label %resume.0.landing

resume.0.landing:                                 ; preds = %resume.0
  br label %AfterCoroSuspend

AfterCoroSuspend:                                 ; preds = %resume.0.landing
  unreachable

resume.entry:                                     ; preds = %entry.resume
  br label %resume.0
}

define internal fastcc void @foo.destroy(ptr noundef nonnull align 8 dereferenceable(24) %0) personality ptr null {
entry.destroy:
  br label %resume.entry

resume.0:                                         ; preds = %resume.entry
  br label %resume.0.landing

resume.0.landing:                                 ; preds = %resume.0
  br label %AfterCoroSuspend

AfterCoroSuspend:                                 ; preds = %resume.0.landing
  unreachable

resume.entry:                                     ; preds = %entry.destroy
  br label %resume.0
}

define internal fastcc void @foo.cleanup(ptr noundef nonnull align 8 dereferenceable(24) %0) personality ptr null {
entry.cleanup:
  br label %resume.entry

resume.0:                                         ; preds = %resume.entry
  br label %resume.0.landing

resume.0.landing:                                 ; preds = %resume.0
  br label %AfterCoroSuspend

AfterCoroSuspend:                                 ; preds = %resume.0.landing
  unreachable

resume.entry:                                     ; preds = %entry.cleanup
  br label %resume.0
}

define internal void @foo.noalloc(ptr %0, ptr %1, ptr noundef nonnull align 8 dereferenceable(24) %2) personality ptr null {
entry:
  %3 = call token @llvm.coro.id(i32 0, ptr null, ptr nonnull @foo, ptr @foo.resumers)
  %resume.addr = getelementptr inbounds nuw %foo.Frame, ptr %2, i32 0, i32 0
  store ptr @foo.resume, ptr %resume.addr, align 8
  %destroy.addr = getelementptr inbounds nuw %foo.Frame, ptr %2, i32 0, i32 1
  store ptr @foo.destroy, ptr %destroy.addr, align 8
  br label %AllocaSpillBB

AllocaSpillBB:                                    ; preds = %entry
  br label %PostSpill

PostSpill:                                        ; preds = %AllocaSpillBB
  br label %CoroSave

CoroSave:                                         ; preds = %PostSpill
  %index.addr1 = getelementptr inbounds nuw %foo.Frame, ptr %2, i32 0, i32 2
  store i1 false, ptr %index.addr1, align 1
  br label %CoroSuspend

CoroSuspend:                                      ; preds = %CoroSave
  br label %resume.0.landing

resume.0.landing:                                 ; preds = %CoroSuspend
  br label %AfterCoroSuspend

AfterCoroSuspend:                                 ; preds = %resume.0.landing
  ret void
}

attributes #0 = { presplitcoroutine }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: read) }
attributes #2 = { nounwind }
attributes #3 = { nomerge nounwind }
attributes #4 = { coro_elide_safe }
