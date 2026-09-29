; RUN: opt -passes=ipsccp -S %s | FileCheck %s
; RUN: opt -passes='coro-early,ipsccp,cgscc(coro-split),verify' -S %s | FileCheck %s --check-prefix=SPLIT

; A returned-continuation coroutine has no ret before splitting. Its
; coro.end is replaced with a real pair of continuation and yield pointers
; during splitting. IPSCCP must not replace the direct call's result with
; undef just because the pre-split body ends in unreachable.

declare token @llvm.coro.id.retcon.once(i32, i32, ptr, ptr, ptr, ptr)
declare ptr @llvm.coro.begin(token, ptr)
declare i1 @llvm.coro.suspend.retcon.i1(...)
declare void @llvm.coro.end(ptr, i1, token)
declare ptr @llvm.coro.prepare.retcon(ptr)
declare ptr @malloc(i64)
declare void @free(ptr)
declare void @resume(ptr, i1)
declare void @consume(ptr)

define internal swiftcc { ptr, ptr } @accessor(ptr noalias %buffer,
                                                ptr swiftself %object) #0 {
entry:
  %id = call token @llvm.coro.id.retcon.once(
      i32 32, i32 8, ptr %buffer, ptr @resume, ptr @malloc, ptr @free)
  %frame = call ptr @llvm.coro.begin(token %id, ptr null)
  %field = getelementptr i8, ptr %object, i64 8
  %suspended = call i1 (...) @llvm.coro.suspend.retcon.i1(ptr %field)
  call void @llvm.coro.end(ptr %frame, i1 false, token none)
  unreachable
}

define void @caller(ptr %target, ptr %buffer, ptr %object) {
entry:
  %prepared = call ptr @llvm.coro.prepare.retcon(ptr %target)
  %is_direct = icmp eq ptr %prepared, @accessor
  br i1 %is_direct, label %direct, label %indirect

direct:
  %direct_pair = call swiftcc { ptr, ptr } @accessor(
      ptr noalias %buffer, ptr swiftself %object)
  br label %join

indirect:
  %indirect_pair = call swiftcc { ptr, ptr } %prepared(
      ptr noalias %buffer, ptr swiftself %object)
  br label %join

join:
  %pair = phi { ptr, ptr } [ %direct_pair, %direct ],
                          [ %indirect_pair, %indirect ]
  %continuation = extractvalue { ptr, ptr } %pair, 0
  %field = extractvalue { ptr, ptr } %pair, 1
  call void @consume(ptr %field)
  call swiftcc void %continuation(ptr %buffer, i1 false)
  ret void
}

; CHECK-LABEL: define void @caller(
; CHECK: direct:
; CHECK: %direct_pair = call swiftcc { ptr, ptr } @accessor(
; CHECK: join:
; CHECK: %pair = phi { ptr, ptr } [ %direct_pair, %direct ], [ %indirect_pair, %indirect ]

; SPLIT-LABEL: define internal swiftcc { ptr, ptr } @accessor(
; SPLIT: ret { ptr, ptr }
; SPLIT-LABEL: define void @caller(
; SPLIT: %pair = phi { ptr, ptr } [ %direct_pair, %direct ], [ %indirect_pair, %indirect ]

define internal i32 @ordinary() {
entry:
  ret i32 7
}

define i32 @ordinary_caller() {
entry:
  %value = call i32 @ordinary()
  ret i32 %value
}

; CHECK-LABEL: define i32 @ordinary_caller()
; CHECK: ret i32 7

attributes #0 = { noinline presplitcoroutine }
