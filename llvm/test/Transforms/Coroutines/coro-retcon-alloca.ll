; RUN: opt < %s -enable-coroutines -O2 -S | FileCheck %s

target datalayout = "p:64:64:64"

declare {i8*, i8*, i32} @prototype_f(i8*, i1)
define {i8*, i8*, i32} @f(i8* %buffer, i32 %n) {
entry:
  %id = call token @llvm.coro.id.retcon(i32 1024, i32 8, i8* %buffer, i8* bitcast ({i8*, i8*, i32} (i8*, i1)* @prototype_f to i8*), i8* bitcast (i8* (i32)* @allocate to i8*), i8* bitcast (void (i8*)* @deallocate to i8*))
  %hdl = call i8* @llvm.coro.begin(token %id, i8* null)
  br label %loop

loop:
  %n.val = phi i32 [ %n, %entry ], [ %inc, %resume ]
  %alloca = call token @llvm.coro.alloca.alloc.i32(i32 %n.val, i32 8)
  %ptr = call i8* @llvm.coro.alloca.get(token %alloca)
  %unwind = call i1 (...) @llvm.coro.suspend.retcon.i1(i8* %ptr, i32 %n.val)
  call void @llvm.coro.alloca.free(token %alloca)
  br i1 %unwind, label %cleanup, label %resume

resume:
  %inc = add i32 %n.val, 1
  br label %loop

cleanup:
  call i1 @llvm.coro.end(i8* %hdl, i1 0)
  unreachable
}

; CHECK-LABEL: define { i8*, i8*, i32 } @f(i8* %buffer, i32 %n)
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[T0:%.*]] = getelementptr inbounds i8, i8* %buffer, i64 8
; CHECK-NEXT:    [[T1:%.*]] = bitcast i8* [[T0]] to i32*
; CHECK-NEXT:    store i32 %n, i32* [[T1]], align 4
; CHECK-NEXT:    [[ALLOC:%.*]] = tail call i8* @allocate(i32 %n)
; CHECK-NEXT:    [[T1:%.*]] = bitcast i8* %buffer to i8**
; CHECK-NEXT:    store i8* [[ALLOC]], i8** [[T1]], align 8
; CHECK-NEXT:    [[T0:%.*]] = insertvalue { i8*, i8*, i32 } { i8* bitcast ({ i8*, i8*, i32 } (i8*, i1)* @f.resume.0 to i8*), i8* undef, i32 undef }, i8* [[ALLOC]], 1
; CHECK-NEXT:    [[RET:%.*]] = insertvalue { i8*, i8*, i32 } [[T0]], i32 %n, 2
; CHECK-NEXT:    ret { i8*, i8*, i32 } [[RET]]
; CHECK-NEXT:  }

; CHECK-LABEL: define internal { i8*, i8*, i32 } @f.resume.0(i8* noalias nonnull align 8 dereferenceable(1024) %0, i1 %1)
; CHECK-NEXT:  :
; CHECK-NEXT:    [[T1:%.*]] = bitcast i8* %0 to i8**
; CHECK-NEXT:    [[ALLOC:%.*]] = load i8*, i8** [[T1]], align 8
; CHECK-NEXT:    tail call void @deallocate(i8* [[ALLOC]])
; CHECK-NEXT:    br i1 %1,

declare {i8*, i32} @prototype_g(i8*, i1)
define {i8*, i32} @g(i8* %buffer, i32 %n) {
entry:
  %id = call token @llvm.coro.id.retcon(i32 1024, i32 8, i8* %buffer, i8* bitcast ({i8*, i32} (i8*, i1)* @prototype_g to i8*), i8* bitcast (i8* (i32)* @allocate to i8*), i8* bitcast (void (i8*)* @deallocate to i8*))
  %hdl = call i8* @llvm.coro.begin(token %id, i8* null)
  br label %loop

loop:
  %n.val = phi i32 [ %n, %entry ], [ %inc, %resume ]
  %alloca = call token @llvm.coro.alloca.alloc.i32(i32 %n.val, i32 8)
  %ptr = call i8* @llvm.coro.alloca.get(token %alloca)
  call void @use(i8* %ptr)
  call void @llvm.coro.alloca.free(token %alloca)
  %unwind = call i1 (...) @llvm.coro.suspend.retcon.i1(i32 %n.val)
  br i1 %unwind, label %cleanup, label %resume

resume:
  %inc = add i32 %n.val, 1
  br label %loop

cleanup:
  call i1 @llvm.coro.end(i8* %hdl, i1 0)
  unreachable
}

; CHECK-LABEL: define { i8*, i32 } @g(i8* %buffer, i32 %n)
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[T0:%.*]] = bitcast i8* %buffer to i32*
; CHECK-NEXT:    store i32 %n, i32* [[T0]], align 4
; CHECK-NEXT:    [[T0:%.*]] = zext i32 %n to i64
; CHECK-NEXT:    [[ALLOC:%.*]] = alloca i8, i64 [[T0]], align 8
; CHECK-NEXT:    call void @use(i8* nonnull [[ALLOC]])
; CHECK-NEXT:    [[RET:%.*]] = insertvalue { i8*, i32 } { i8* bitcast ({ i8*, i32 } (i8*, i1)* @g.resume.0 to i8*), i32 undef }, i32 %n, 1
; CHECK-NEXT:    ret { i8*, i32 } [[RET]]
; CHECK-NEXT:  }

; CHECK-LABEL: define internal { i8*, i32 } @g.resume.0(i8* noalias nonnull align 8 dereferenceable(1024) %0, i1 %1)
; CHECK-NEXT:  :
; CHECK-NEXT:    br i1 %1,
; CHECK:       :
; CHECK-NEXT:    [[T0:%.*]] = bitcast i8* %0 to i32*
; CHECK-NEXT:    [[T1:%.*]] = load i32, i32* [[T0]], align 8
; CHECK-NEXT:    %inc = add i32 [[T1]], 1
; CHECK-NEXT:    store i32 %inc, i32* [[T0]], align 8
; CHECK-NEXT:    [[T0:%.*]] = zext i32 %inc to i64
; CHECK-NEXT:    [[ALLOC:%.*]] = alloca i8, i64 [[T0]], align 8
; CHECK-NEXT:    call void @use(i8* nonnull [[ALLOC]])
; CHECK-NEXT:    [[RET:%.*]] = insertvalue { i8*, i32 } { i8* bitcast ({ i8*, i32 } (i8*, i1)* @g.resume.0 to i8*), i32 undef }, i32 %inc, 1
; CHECK-NEXT:    ret { i8*, i32 } [[RET]]
; CHECK:       :
; CHECK-NEXT:    ret { i8*, i32 } { i8* null, i32 undef }

declare {i8*, i32} @prototype_h(i8*, i1)
define {i8*, i32} @h(i8* %buffer, i32 %n) {
entry:
  %id = call token @llvm.coro.id.retcon(i32 1024, i32 8, i8* %buffer, i8* bitcast ({i8*, i32} (i8*, i1)* @prototype_h to i8*), i8* bitcast (i8* (i32)* @allocate to i8*), i8* bitcast (void (i8*)* @deallocate to i8*))
  %hdl = call i8* @llvm.coro.begin(token %id, i8* null)
  br label %loop

loop:
  %n.val = phi i32 [ %n, %entry ], [ %inc, %resume ]
  %unwind = call i1 (...) @llvm.coro.suspend.retcon.i1(i32 %n.val)
  br i1 %unwind, label %cleanup, label %resume

resume:
  %inc = add i32 %n.val, 1
  %alloca = call token @llvm.coro.alloca.alloc.i32(i32 %inc, i32 8)
  %ptr = call i8* @llvm.coro.alloca.get(token %alloca)
  call void @use(i8* %ptr)
  call void @llvm.coro.alloca.free(token %alloca)
  br label %loop

cleanup:
  call i1 @llvm.coro.end(i8* %hdl, i1 0)
  unreachable
}

; CHECK-LABEL: define { i8*, i32 } @h(i8* %buffer, i32 %n)
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[T0:%.*]] = bitcast i8* %buffer to i32*
; CHECK-NEXT:    store i32 %n, i32* [[T0]], align 4
; CHECK-NEXT:    [[RET:%.*]] = insertvalue { i8*, i32 } { i8* bitcast ({ i8*, i32 } (i8*, i1)* @h.resume.0 to i8*), i32 undef }, i32 %n, 1
; CHECK-NEXT:    ret { i8*, i32 } [[RET]]
; CHECK-NEXT:  }

; CHECK-LABEL: define internal { i8*, i32 } @h.resume.0(i8* noalias nonnull align 8 dereferenceable(1024) %0, i1 %1)
; CHECK-NEXT:  :
; CHECK-NEXT:    br i1 %1,
; CHECK:       :
; CHECK-NEXT:    [[NSLOT:%.*]] = bitcast i8* %0 to i32*
; CHECK-NEXT:    [[T1:%.*]] = load i32, i32* [[NSLOT]], align 8
; CHECK-NEXT:    %inc = add i32 [[T1]], 1
; CHECK-NEXT:    [[T0:%.*]] = zext i32 %inc to i64
; CHECK-NEXT:    [[ALLOC:%.*]] = alloca i8, i64 [[T0]], align 8
; CHECK-NEXT:    call void @use(i8* nonnull [[ALLOC]])
; CHECK-NEXT:    store i32 %inc, i32* [[NSLOT]], align 8
; CHECK-NEXT:    [[RET:%.*]] = insertvalue { i8*, i32 } { i8* bitcast ({ i8*, i32 } (i8*, i1)* @h.resume.0 to i8*), i32 undef }, i32 %inc, 1
; CHECK-NEXT:    ret { i8*, i32 } [[RET]]
; CHECK:       :
; CHECK-NEXT:    ret { i8*, i32 } { i8* null, i32 undef }

declare {i8*, i32} @prototype_i(i8*)
define {i8*, i32} @i(i8* %buffer, i32 %n) {
entry:
  %id = call token @llvm.coro.id.retcon(i32 1024, i32 8, i8* %buffer, i8* bitcast ({i8*, i32} (i8*)* @prototype_i to i8*), i8* bitcast (i8* (i32)* @allocate to i8*), i8* bitcast (void (i8*)* @deallocate to i8*))
  %hdl = call i8* @llvm.coro.begin(token %id, i8* null)
  br label %loop

loop:
  %n.val = phi i32 [ %n, %entry ], [ %k, %loop2 ]
  call void (...) @llvm.coro.suspend.retcon.isVoid(i32 %n.val)
  %inc = add i32 %n.val, 1
  br label %loop2

loop2:
  %k = phi i32 [ %inc, %loop ], [ %k2, %loop2 ]
  %alloca = call token @llvm.coro.alloca.alloc.i32(i32 %k, i32 8)
  %ptr = call i8* @llvm.coro.alloca.get(token %alloca)
  call void @use(i8* %ptr)
  call void @llvm.coro.alloca.free(token %alloca)
  %k2 = lshr i32 %k, 1
  %cmp = icmp ugt i32 %k, 128
  br i1 %cmp, label %loop2, label %loop
}

; CHECK-LABEL: define { i8*, i32 } @i(i8* %buffer, i32 %n)
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[T0:%.*]] = bitcast i8* %buffer to i32*
; CHECK-NEXT:    store i32 %n, i32* [[T0]], align 4
; CHECK-NEXT:    [[RET:%.*]] = insertvalue { i8*, i32 } { i8* bitcast ({ i8*, i32 } (i8*)* @i.resume.0 to i8*), i32 undef }, i32 %n, 1
; CHECK-NEXT:    ret { i8*, i32 } [[RET]]
; CHECK-NEXT:  }

; CHECK-LABEL: define internal { i8*, i32 } @i.resume.0(i8* noalias nonnull align 8 dereferenceable(1024) %0)
; CHECK-NEXT:  :
; CHECK-NEXT:    [[NSLOT:%.*]] = bitcast i8* %0 to i32*
; CHECK-NEXT:    [[T1:%.*]] = load i32, i32* [[NSLOT]], align 8
; CHECK-NEXT:    %inc = add i32 [[T1]], 1
; CHECK-NEXT:    br label %loop2
; CHECK:       :
; CHECK-NEXT:    store i32 %k, i32* [[NSLOT]], align 8
; CHECK-NEXT:    [[RET:%.*]] = insertvalue { i8*, i32 } { i8* bitcast ({ i8*, i32 } (i8*)* @i.resume.0 to i8*), i32 undef }, i32 %k, 1
; CHECK-NEXT:    ret { i8*, i32 } [[RET]]
; CHECK:       loop2:
; CHECK-NEXT:    %k = phi i32 [ %inc, {{.*}} ], [ %k2, %loop2 ]
; CHECK-NEXT:    [[SAVE:%.*]] = call i8* @llvm.stacksave()
; CHECK-NEXT:    [[T0:%.*]] = zext i32 %k to i64
; CHECK-NEXT:    [[ALLOC:%.*]] = alloca i8, i64 [[T0]], align 8
; CHECK-NEXT:    call void @use(i8* nonnull [[ALLOC]])
; CHECK-NEXT:    call void @llvm.stackrestore(i8* [[SAVE]])
; CHECK-NEXT:    %k2 = lshr i32 %k, 1
; CHECK-NEXT:    %cmp = icmp ugt i32 %k, 128
; CHECK-NEXT:    br i1 %cmp, label %loop2,
; CHECK-NEXT:  }

declare {i8*, i32} @prototype_j(i8*)
define {i8*, i32} @j(i8* %buffer, i32 %n) {
entry:
  %id = call token @llvm.coro.id.retcon(i32 1024, i32 8, i8* %buffer, i8* bitcast ({i8*, i32} (i8*)* @prototype_j to i8*), i8* bitcast (i8* (i32)* @allocate to i8*), i8* bitcast (void (i8*)* @deallocate to i8*))
  %hdl = call i8* @llvm.coro.begin(token %id, i8* null)
  br label %forward

back:
  ; We should encounter this 'get' before we encounter the 'alloc'.
  %ptr = call i8* @llvm.coro.alloca.get(token %alloca)
  call void @use(i8* %ptr)
  call void @llvm.coro.alloca.free(token %alloca)
  %k = add i32 %n.val, 1
  %cmp = icmp ugt i32 %k, 128
  br i1 %cmp, label %forward, label %end

forward:
  %n.val = phi i32 [ %n, %entry ], [ %k, %back ]
  call void (...) @llvm.coro.suspend.retcon.isVoid(i32 %n.val)
  %alloca = call token @llvm.coro.alloca.alloc.i32(i32 %n.val, i32 8)
  %inc = add i32 %n.val, 1
  br label %back

end:
  call i1 @llvm.coro.end(i8* %hdl, i1 0)
  unreachable
}

declare token @llvm.coro.id.retcon(i32, i32, i8*, i8*, i8*, i8*)
declare i8* @llvm.coro.begin(token, i8*)
declare i1 @llvm.coro.suspend.retcon.i1(...)
declare void @llvm.coro.suspend.retcon.isVoid(...)
declare i1 @llvm.coro.end(i8*, i1)
declare i8* @llvm.coro.prepare.retcon(i8*)
declare token @llvm.coro.alloca.alloc.i32(i32, i32)
declare i8* @llvm.coro.alloca.get(token)
declare void @llvm.coro.alloca.free(token)

declare noalias i8* @allocate(i32 %size)
declare void @deallocate(i8* %ptr)

declare void @print(i32)
declare void @use(i8*)
