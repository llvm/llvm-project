; RUN: llvm-as -disable-output < %s

%"struct.std::pair<int,int>" = type { i32, i32 }

declare void @_Z3barRKi(ptr)

declare void @llvm.lifetime.start(i64, ptr nocapture) nounwind
declare void @llvm.lifetime.end(i64, ptr nocapture) nounwind
declare ptr @llvm.invariant.start.p0(i64, ptr nocapture) readonly nounwind
declare void @llvm.invariant.end.p0(ptr, i64, ptr nocapture) nounwind

define i32 @_Z4foo2v() nounwind {
entry:
  %x = alloca %"struct.std::pair<int,int>"

  ;; Constructor starts here (this isn't needed since it is immediately
  ;; preceded by an alloca, but shown for completeness).
  call void @llvm.lifetime.start(i64 8, ptr %x)

  %0 = getelementptr %"struct.std::pair<int,int>", ptr %x, i32 0, i32 0
  store i32 4, ptr %0, align 8
  %1 = getelementptr %"struct.std::pair<int,int>", ptr %x, i32 0, i32 1
  store i32 5, ptr %1, align 4

  ;; Constructor has finished here.
  %inv = call ptr @llvm.invariant.start.p0(i64 8, ptr %x)
  call void @_Z3barRKi(ptr %0) nounwind
  %2 = load i32, ptr %0, align 8

  ;; Destructor is run here.
  call void @llvm.invariant.end.p0(ptr %inv, i64 8, ptr %x)
  ;; Destructor is done here.
  call void @llvm.lifetime.end(i64 8, ptr %x)
  ret i32 %2
}
