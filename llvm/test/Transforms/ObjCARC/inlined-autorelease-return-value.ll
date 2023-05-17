; RUN: opt -passes=objc-arc -S < %s | FileCheck %s

target datalayout = "e-p:64:64:64"

declare ptr @llvm.objc.retain(ptr)
declare ptr @llvm.objc.autoreleaseReturnValue(ptr)
declare ptr @llvm.objc.retainAutoreleasedReturnValue(ptr)
declare ptr @llvm.objc.unsafeClaimAutoreleasedReturnValue(ptr)
declare void @opaque()
declare void @llvm.lifetime.start(i64, ptr nocapture)
declare void @llvm.lifetime.end(i64, ptr nocapture)

; CHECK-LABEL: define ptr @elide_with_retainRV(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    ret ptr %x
define ptr @elide_with_retainRV(ptr %x) nounwind {
entry:
  %b = call ptr @llvm.objc.autoreleaseReturnValue(ptr %x) nounwind
  %c = call ptr @llvm.objc.retainAutoreleasedReturnValue(ptr %b) nounwind
  ret ptr %c
}

; CHECK-LABEL: define ptr @elide_with_retainRV_bitcast(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    ret ptr %x
define ptr @elide_with_retainRV_bitcast(ptr %x) nounwind {
entry:
  %b = call ptr @llvm.objc.autoreleaseReturnValue(ptr %x) nounwind
  %d = call ptr @llvm.objc.retainAutoreleasedReturnValue(ptr %x) nounwind
  ret ptr %d
}

; CHECK-LABEL: define ptr @elide_with_retainRV_phi(
; CHECK-NOT:   define
; CHECK:       phis:
; CHECK-NEXT:    phi ptr
; CHECK-NEXT:    ret ptr
define ptr @elide_with_retainRV_phi(ptr %x) nounwind {
entry:
  br label %phis

phis:
  %a = phi ptr [ %x, %entry ]
  %c = phi ptr [ %x, %entry ]
  %b = call ptr @llvm.objc.autoreleaseReturnValue(ptr %a) nounwind
  %d = call ptr @llvm.objc.retainAutoreleasedReturnValue(ptr %c) nounwind
  ret ptr %d
}

; CHECK-LABEL: define ptr @elide_with_retainRV_splitByRetain(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    %b = call ptr @llvm.objc.autorelease(ptr %x)
; CHECK-NEXT:    tail call ptr @llvm.objc.retain(ptr %x)
; CHECK-NEXT:    tail call ptr @llvm.objc.retain(ptr %b)
define ptr @elide_with_retainRV_splitByRetain(ptr %x) nounwind {
entry:
  ; Cleanup is blocked by other ARC intrinsics for ease of implementation; we
  ; only delay processing AutoreleaseRV until the very next ARC intrinsic.  In
  ; practice, it would be very strange for this to matter.
  %b = call ptr @llvm.objc.autoreleaseReturnValue(ptr %x) nounwind
  %c = call ptr @llvm.objc.retain(ptr %x) nounwind
  %d = call ptr @llvm.objc.retainAutoreleasedReturnValue(ptr %b) nounwind
  ret ptr %d
}

; CHECK-LABEL: define ptr @elide_with_retainRV_splitByOpaque(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    %b = call ptr @llvm.objc.autorelease(ptr %x)
; CHECK-NEXT:    call void @opaque()
; CHECK-NEXT:    %d = tail call ptr @llvm.objc.retain(ptr %b)
; CHECK-NEXT:    ret ptr %d
define ptr @elide_with_retainRV_splitByOpaque(ptr %x) nounwind {
entry:
  ; Cleanup should get blocked by opaque calls.
  %b = call ptr @llvm.objc.autoreleaseReturnValue(ptr %x) nounwind
  call void @opaque() nounwind
  %d = call ptr @llvm.objc.retainAutoreleasedReturnValue(ptr %b) nounwind
  ret ptr %d
}

; CHECK-LABEL: define ptr @elide_with_retainRV_splitByLifetime(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    call void @llvm.lifetime.start.p0(i64 8, ptr %x)
; CHECK-NEXT:    call void @llvm.lifetime.end.p0(i64 8, ptr %x)
; CHECK-NEXT:    ret ptr %x
define ptr @elide_with_retainRV_splitByLifetime(ptr %x) nounwind {
entry:
  ; Cleanup should skip over lifetime intrinsics.
  call void @llvm.lifetime.start(i64 8, ptr %x)
  %b = call ptr @llvm.objc.autoreleaseReturnValue(ptr %x) nounwind
  call void @llvm.lifetime.end(i64 8, ptr %x)
  %d = call ptr @llvm.objc.retainAutoreleasedReturnValue(ptr %b) nounwind
  ret ptr %d
}

; CHECK-LABEL: define ptr @elide_with_retainRV_wrongArg(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    call void @llvm.objc.release(ptr %x)
; CHECK-NEXT:    tail call ptr @llvm.objc.retain(ptr %y)
define ptr @elide_with_retainRV_wrongArg(ptr %x, ptr %y) nounwind {
entry:
  %b = call ptr @llvm.objc.autoreleaseReturnValue(ptr %x) nounwind
  %c = call ptr @llvm.objc.retainAutoreleasedReturnValue(ptr %y) nounwind
  ret ptr %c
}

; CHECK-LABEL: define ptr @elide_with_retainRV_wrongBB(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    call ptr @llvm.objc.autorelease(ptr %x)
; CHECK-NEXT:    br label %next
; CHECK:       next:
; CHECK-NEXT:    tail call ptr @llvm.objc.retain(
; CHECK-NEXT:    ret ptr
define ptr @elide_with_retainRV_wrongBB(ptr %x) nounwind {
entry:
  %b = call ptr @llvm.objc.autoreleaseReturnValue(ptr %x) nounwind
  br label %next

next:
  %c = call ptr @llvm.objc.retainAutoreleasedReturnValue(ptr %b) nounwind
  ret ptr %c
}

; CHECK-LABEL: define ptr @elide_with_retainRV_beforeAutoreleaseRV(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    tail call ptr @llvm.objc.autoreleaseReturnValue(ptr %x)
; CHECK-NEXT:    ret ptr %x
define ptr @elide_with_retainRV_beforeAutoreleaseRV(ptr %x) nounwind {
entry:
  %b = call ptr @llvm.objc.autoreleaseReturnValue(ptr %x) nounwind
  %c = call ptr @llvm.objc.retainAutoreleasedReturnValue(ptr %b) nounwind
  %d = call ptr @llvm.objc.autoreleaseReturnValue(ptr %c) nounwind
  ret ptr %c
}

; CHECK-LABEL: define ptr @elide_with_retainRV_afterRetain(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    tail call ptr @llvm.objc.retain(ptr %x)
; CHECK-NEXT:    ret ptr %a
define ptr @elide_with_retainRV_afterRetain(ptr %x) nounwind {
entry:
  %a = call ptr @llvm.objc.retain(ptr %x) nounwind
  %b = call ptr @llvm.objc.autoreleaseReturnValue(ptr %a) nounwind
  %c = call ptr @llvm.objc.retainAutoreleasedReturnValue(ptr %b) nounwind
  ret ptr %c
}

; CHECK-LABEL: define ptr @elide_with_claimRV(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    tail call void @llvm.objc.release(ptr %x)
; CHECK-NEXT:    ret ptr %x
define ptr @elide_with_claimRV(ptr %x) nounwind {
entry:
  %b = call ptr @llvm.objc.autoreleaseReturnValue(ptr %x) nounwind
  %c = call ptr @llvm.objc.unsafeClaimAutoreleasedReturnValue(ptr %b) nounwind
  ret ptr %c
}

; CHECK-LABEL: define ptr @elide_with_claimRV_bitcast(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    tail call void @llvm.objc.release(ptr %x)
; CHECK-NEXT:    ret ptr %x
define ptr @elide_with_claimRV_bitcast(ptr %x) nounwind {
entry:
  %b = call ptr @llvm.objc.autoreleaseReturnValue(ptr %x) nounwind
  %d = call ptr @llvm.objc.unsafeClaimAutoreleasedReturnValue(ptr %x) nounwind
  ret ptr %d
}

; CHECK-LABEL: define ptr @elide_with_claimRV_phi(
; CHECK-NOT:   define
; CHECK:       phis:
; CHECK-NEXT:    %c = phi ptr
; CHECK-NEXT:    tail call void @llvm.objc.release(ptr %c)
; CHECK-NEXT:    ret ptr %c
define ptr @elide_with_claimRV_phi(ptr %x) nounwind {
entry:
  br label %phis

phis:
  %a = phi ptr [ %x, %entry ]
  %c = phi ptr [ %x, %entry ]
  %b = call ptr @llvm.objc.autoreleaseReturnValue(ptr %a) nounwind
  %d = call ptr @llvm.objc.unsafeClaimAutoreleasedReturnValue(ptr %c) nounwind
  ret ptr %d
}

; CHECK-LABEL: define ptr @elide_with_claimRV_splitByRetain(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    %b = call ptr @llvm.objc.autorelease(ptr %x)
; CHECK-NEXT:    tail call ptr @llvm.objc.retain(ptr %x)
; CHECK-NEXT:    tail call ptr @llvm.objc.unsafeClaimAutoreleasedReturnValue(ptr %b)
define ptr @elide_with_claimRV_splitByRetain(ptr %x) nounwind {
entry:
  ; Cleanup is blocked by other ARC intrinsics for ease of implementation; we
  ; only delay processing AutoreleaseRV until the very next ARC intrinsic.  In
  ; practice, it would be very strange for this to matter.
  %b = call ptr @llvm.objc.autoreleaseReturnValue(ptr %x) nounwind
  %c = call ptr @llvm.objc.retain(ptr %x) nounwind
  %d = call ptr @llvm.objc.unsafeClaimAutoreleasedReturnValue(ptr %b) nounwind
  ret ptr %d
}

; CHECK-LABEL: define ptr @elide_with_claimRV_splitByOpaque(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    %b = call ptr @llvm.objc.autorelease(ptr %x)
; CHECK-NEXT:    call void @opaque()
; CHECK-NEXT:    %d = tail call ptr @llvm.objc.unsafeClaimAutoreleasedReturnValue(ptr %b)
; CHECK-NEXT:    ret ptr %d
define ptr @elide_with_claimRV_splitByOpaque(ptr %x) nounwind {
entry:
  ; Cleanup should get blocked by opaque calls.
  %b = call ptr @llvm.objc.autoreleaseReturnValue(ptr %x) nounwind
  call void @opaque() nounwind
  %d = call ptr @llvm.objc.unsafeClaimAutoreleasedReturnValue(ptr %b) nounwind
  ret ptr %d
}

; CHECK-LABEL: define ptr @elide_with_claimRV_splitByLifetime(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    call void @llvm.lifetime.start.p0(i64 8, ptr %x)
; CHECK-NEXT:    call void @llvm.lifetime.end.p0(i64 8, ptr %x)
; CHECK-NEXT:    tail call void @llvm.objc.release(ptr %x)
; CHECK-NEXT:    ret ptr %x
define ptr @elide_with_claimRV_splitByLifetime(ptr %x) nounwind {
entry:
  ; Cleanup should skip over lifetime intrinsics.
  call void @llvm.lifetime.start(i64 8, ptr %x)
  %b = call ptr @llvm.objc.autoreleaseReturnValue(ptr %x) nounwind
  call void @llvm.lifetime.end(i64 8, ptr %x)
  %d = call ptr @llvm.objc.unsafeClaimAutoreleasedReturnValue(ptr %b) nounwind
  ret ptr %d
}

; CHECK-LABEL: define ptr @elide_with_claimRV_wrongArg(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    call void @llvm.objc.release(ptr %x)
; CHECK-NEXT:    tail call ptr @llvm.objc.unsafeClaimAutoreleasedReturnValue(ptr %y)
define ptr @elide_with_claimRV_wrongArg(ptr %x, ptr %y) nounwind {
entry:
  %b = call ptr @llvm.objc.autoreleaseReturnValue(ptr %x) nounwind
  %c = call ptr @llvm.objc.unsafeClaimAutoreleasedReturnValue(ptr %y) nounwind
  ret ptr %c
}

; CHECK-LABEL: define ptr @elide_with_claimRV_wrongBB(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    call ptr @llvm.objc.autorelease(ptr %x)
; CHECK-NEXT:    br label %next
; CHECK:       next:
; CHECK-NEXT:    tail call ptr @llvm.objc.unsafeClaimAutoreleasedReturnValue(
; CHECK-NEXT:    ret ptr
define ptr @elide_with_claimRV_wrongBB(ptr %x) nounwind {
entry:
  %b = call ptr @llvm.objc.autoreleaseReturnValue(ptr %x) nounwind
  br label %next

next:
  %c = call ptr @llvm.objc.unsafeClaimAutoreleasedReturnValue(ptr %b) nounwind
  ret ptr %c
}


; CHECK-LABEL: define ptr @elide_with_claimRV_beforeAutoreleaseRV(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    tail call void @llvm.objc.release(ptr %x)
; CHECK-NEXT:    tail call ptr @llvm.objc.autoreleaseReturnValue(ptr %x)
; CHECK-NEXT:    ret ptr %x
define ptr @elide_with_claimRV_beforeAutoreleaseRV(ptr %x) nounwind {
entry:
  %b = call ptr @llvm.objc.autoreleaseReturnValue(ptr %x) nounwind
  %c = call ptr @llvm.objc.unsafeClaimAutoreleasedReturnValue(ptr %b) nounwind
  %d = call ptr @llvm.objc.autoreleaseReturnValue(ptr %c) nounwind
  ret ptr %c
}

; CHECK-LABEL: define ptr @elide_with_claimRV_afterRetain(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    ret ptr %x
define ptr @elide_with_claimRV_afterRetain(ptr %x) nounwind {
entry:
  %a = call ptr @llvm.objc.retain(ptr %x) nounwind
  %b = call ptr @llvm.objc.autoreleaseReturnValue(ptr %a) nounwind
  %c = call ptr @llvm.objc.unsafeClaimAutoreleasedReturnValue(ptr %b) nounwind
  ret ptr %c
}
