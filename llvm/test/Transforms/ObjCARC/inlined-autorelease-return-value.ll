; RUN: opt -objc-arc -S < %s | FileCheck %s

target datalayout = "e-p:64:64:64"

declare i8* @llvm.objc.retain(i8*)
declare i8* @llvm.objc.autoreleaseReturnValue(i8*)
declare i8* @llvm.objc.retainAutoreleasedReturnValue(i8*)
declare i8* @llvm.objc.unsafeClaimAutoreleasedReturnValue(i8*)
declare void @opaque()
declare void @llvm.lifetime.start(i64, i8* nocapture)
declare void @llvm.lifetime.end(i64, i8* nocapture)

; CHECK-LABEL: define i8* @elide_with_retainRV(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    ret i8* %x
define i8* @elide_with_retainRV(i8* %x) nounwind {
entry:
  %b = call i8* @llvm.objc.autoreleaseReturnValue(i8* %x) nounwind
  %c = call i8* @llvm.objc.retainAutoreleasedReturnValue(i8* %b) nounwind
  ret i8* %c
}

; CHECK-LABEL: define i8* @elide_with_retainRV_bitcast(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    %c = bitcast i32* %x to i8*
; CHECK-NEXT:    ret i8* %c
define i8* @elide_with_retainRV_bitcast(i32* %x) nounwind {
entry:
  %a = bitcast i32* %x to i8*
  %b = call i8* @llvm.objc.autoreleaseReturnValue(i8* %a) nounwind
  %c = bitcast i32* %x to i8*
  %d = call i8* @llvm.objc.retainAutoreleasedReturnValue(i8* %c) nounwind
  ret i8* %d
}

; CHECK-LABEL: define i8* @elide_with_retainRV_phi(
; CHECK-NOT:   define
; CHECK:       phis:
; CHECK-NEXT:    phi i8*
; CHECK-NEXT:    ret i8*
define i8* @elide_with_retainRV_phi(i8* %x) nounwind {
entry:
  br label %phis

phis:
  %a = phi i8* [ %x, %entry ]
  %c = phi i8* [ %x, %entry ]
  %b = call i8* @llvm.objc.autoreleaseReturnValue(i8* %a) nounwind
  %d = call i8* @llvm.objc.retainAutoreleasedReturnValue(i8* %c) nounwind
  ret i8* %d
}

; CHECK-LABEL: define i8* @elide_with_retainRV_splitByRetain(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    %b = call i8* @llvm.objc.autorelease(i8* %x)
; CHECK-NEXT:    tail call i8* @llvm.objc.retain(i8* %x)
; CHECK-NEXT:    tail call i8* @llvm.objc.retain(i8* %b)
define i8* @elide_with_retainRV_splitByRetain(i8* %x) nounwind {
entry:
  ; Cleanup is blocked by other ARC intrinsics for ease of implementation; we
  ; only delay processing AutoreleaseRV until the very next ARC intrinsic.  In
  ; practice, it would be very strange for this to matter.
  %b = call i8* @llvm.objc.autoreleaseReturnValue(i8* %x) nounwind
  %c = call i8* @llvm.objc.retain(i8* %x) nounwind
  %d = call i8* @llvm.objc.retainAutoreleasedReturnValue(i8* %b) nounwind
  ret i8* %d
}

; CHECK-LABEL: define i8* @elide_with_retainRV_splitByOpaque(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    %b = call i8* @llvm.objc.autorelease(i8* %x)
; CHECK-NEXT:    call void @opaque()
; CHECK-NEXT:    %d = tail call i8* @llvm.objc.retain(i8* %b)
; CHECK-NEXT:    ret i8* %d
define i8* @elide_with_retainRV_splitByOpaque(i8* %x) nounwind {
entry:
  ; Cleanup should get blocked by opaque calls.
  %b = call i8* @llvm.objc.autoreleaseReturnValue(i8* %x) nounwind
  call void @opaque() nounwind
  %d = call i8* @llvm.objc.retainAutoreleasedReturnValue(i8* %b) nounwind
  ret i8* %d
}

; CHECK-LABEL: define i8* @elide_with_retainRV_splitByLifetime(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    call void @llvm.lifetime.start.p0i8(i64 8, i8* %x)
; CHECK-NEXT:    call void @llvm.lifetime.end.p0i8(i64 8, i8* %x)
; CHECK-NEXT:    ret i8* %x
define i8* @elide_with_retainRV_splitByLifetime(i8* %x) nounwind {
entry:
  ; Cleanup should skip over lifetime intrinsics.
  call void @llvm.lifetime.start(i64 8, i8* %x)
  %b = call i8* @llvm.objc.autoreleaseReturnValue(i8* %x) nounwind
  call void @llvm.lifetime.end(i64 8, i8* %x)
  %d = call i8* @llvm.objc.retainAutoreleasedReturnValue(i8* %b) nounwind
  ret i8* %d
}

; CHECK-LABEL: define i8* @elide_with_retainRV_wrongArg(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    call void @llvm.objc.release(i8* %x)
; CHECK-NEXT:    tail call i8* @llvm.objc.retain(i8* %y)
define i8* @elide_with_retainRV_wrongArg(i8* %x, i8* %y) nounwind {
entry:
  %b = call i8* @llvm.objc.autoreleaseReturnValue(i8* %x) nounwind
  %c = call i8* @llvm.objc.retainAutoreleasedReturnValue(i8* %y) nounwind
  ret i8* %c
}

; CHECK-LABEL: define i8* @elide_with_retainRV_wrongBB(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    call i8* @llvm.objc.autorelease(i8* %x)
; CHECK-NEXT:    br label %next
; CHECK:       next:
; CHECK-NEXT:    tail call i8* @llvm.objc.retain(
; CHECK-NEXT:    ret i8*
define i8* @elide_with_retainRV_wrongBB(i8* %x) nounwind {
entry:
  %b = call i8* @llvm.objc.autoreleaseReturnValue(i8* %x) nounwind
  br label %next

next:
  %c = call i8* @llvm.objc.retainAutoreleasedReturnValue(i8* %b) nounwind
  ret i8* %c
}

; CHECK-LABEL: define i8* @elide_with_retainRV_beforeAutoreleaseRV(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    tail call i8* @llvm.objc.autoreleaseReturnValue(i8* %x)
; CHECK-NEXT:    ret i8* %x
define i8* @elide_with_retainRV_beforeAutoreleaseRV(i8* %x) nounwind {
entry:
  %b = call i8* @llvm.objc.autoreleaseReturnValue(i8* %x) nounwind
  %c = call i8* @llvm.objc.retainAutoreleasedReturnValue(i8* %b) nounwind
  %d = call i8* @llvm.objc.autoreleaseReturnValue(i8* %c) nounwind
  ret i8* %c
}

; CHECK-LABEL: define i8* @elide_with_retainRV_afterRetain(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    tail call i8* @llvm.objc.retain(i8* %x)
; CHECK-NEXT:    ret i8* %a
define i8* @elide_with_retainRV_afterRetain(i8* %x) nounwind {
entry:
  %a = call i8* @llvm.objc.retain(i8* %x) nounwind
  %b = call i8* @llvm.objc.autoreleaseReturnValue(i8* %a) nounwind
  %c = call i8* @llvm.objc.retainAutoreleasedReturnValue(i8* %b) nounwind
  ret i8* %c
}

; CHECK-LABEL: define i8* @elide_with_claimRV(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    tail call void @llvm.objc.release(i8* %x)
; CHECK-NEXT:    ret i8* %x
define i8* @elide_with_claimRV(i8* %x) nounwind {
entry:
  %b = call i8* @llvm.objc.autoreleaseReturnValue(i8* %x) nounwind
  %c = call i8* @llvm.objc.unsafeClaimAutoreleasedReturnValue(i8* %b) nounwind
  ret i8* %c
}

; CHECK-LABEL: define i8* @elide_with_claimRV_bitcast(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    %c = bitcast i32* %x to i8*
; CHECK-NEXT:    tail call void @llvm.objc.release(i8* %c)
; CHECK-NEXT:    ret i8* %c
define i8* @elide_with_claimRV_bitcast(i32* %x) nounwind {
entry:
  %a = bitcast i32* %x to i8*
  %b = call i8* @llvm.objc.autoreleaseReturnValue(i8* %a) nounwind
  %c = bitcast i32* %x to i8*
  %d = call i8* @llvm.objc.unsafeClaimAutoreleasedReturnValue(i8* %c) nounwind
  ret i8* %d
}

; CHECK-LABEL: define i8* @elide_with_claimRV_phi(
; CHECK-NOT:   define
; CHECK:       phis:
; CHECK-NEXT:    %c = phi i8*
; CHECK-NEXT:    tail call void @llvm.objc.release(i8* %c)
; CHECK-NEXT:    ret i8* %c
define i8* @elide_with_claimRV_phi(i8* %x) nounwind {
entry:
  br label %phis

phis:
  %a = phi i8* [ %x, %entry ]
  %c = phi i8* [ %x, %entry ]
  %b = call i8* @llvm.objc.autoreleaseReturnValue(i8* %a) nounwind
  %d = call i8* @llvm.objc.unsafeClaimAutoreleasedReturnValue(i8* %c) nounwind
  ret i8* %d
}

; CHECK-LABEL: define i8* @elide_with_claimRV_splitByRetain(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    %b = call i8* @llvm.objc.autorelease(i8* %x)
; CHECK-NEXT:    tail call i8* @llvm.objc.retain(i8* %x)
; CHECK-NEXT:    tail call i8* @llvm.objc.unsafeClaimAutoreleasedReturnValue(i8* %b)
define i8* @elide_with_claimRV_splitByRetain(i8* %x) nounwind {
entry:
  ; Cleanup is blocked by other ARC intrinsics for ease of implementation; we
  ; only delay processing AutoreleaseRV until the very next ARC intrinsic.  In
  ; practice, it would be very strange for this to matter.
  %b = call i8* @llvm.objc.autoreleaseReturnValue(i8* %x) nounwind
  %c = call i8* @llvm.objc.retain(i8* %x) nounwind
  %d = call i8* @llvm.objc.unsafeClaimAutoreleasedReturnValue(i8* %b) nounwind
  ret i8* %d
}

; CHECK-LABEL: define i8* @elide_with_claimRV_splitByOpaque(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    %b = call i8* @llvm.objc.autorelease(i8* %x)
; CHECK-NEXT:    call void @opaque()
; CHECK-NEXT:    %d = tail call i8* @llvm.objc.unsafeClaimAutoreleasedReturnValue(i8* %b)
; CHECK-NEXT:    ret i8* %d
define i8* @elide_with_claimRV_splitByOpaque(i8* %x) nounwind {
entry:
  ; Cleanup should get blocked by opaque calls.
  %b = call i8* @llvm.objc.autoreleaseReturnValue(i8* %x) nounwind
  call void @opaque() nounwind
  %d = call i8* @llvm.objc.unsafeClaimAutoreleasedReturnValue(i8* %b) nounwind
  ret i8* %d
}

; CHECK-LABEL: define i8* @elide_with_claimRV_splitByLifetime(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    call void @llvm.lifetime.start.p0i8(i64 8, i8* %x)
; CHECK-NEXT:    call void @llvm.lifetime.end.p0i8(i64 8, i8* %x)
; CHECK-NEXT:    tail call void @llvm.objc.release(i8* %x)
; CHECK-NEXT:    ret i8* %x
define i8* @elide_with_claimRV_splitByLifetime(i8* %x) nounwind {
entry:
  ; Cleanup should skip over lifetime intrinsics.
  call void @llvm.lifetime.start(i64 8, i8* %x)
  %b = call i8* @llvm.objc.autoreleaseReturnValue(i8* %x) nounwind
  call void @llvm.lifetime.end(i64 8, i8* %x)
  %d = call i8* @llvm.objc.unsafeClaimAutoreleasedReturnValue(i8* %b) nounwind
  ret i8* %d
}

; CHECK-LABEL: define i8* @elide_with_claimRV_wrongArg(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    call void @llvm.objc.release(i8* %x)
; CHECK-NEXT:    tail call i8* @llvm.objc.unsafeClaimAutoreleasedReturnValue(i8* %y)
define i8* @elide_with_claimRV_wrongArg(i8* %x, i8* %y) nounwind {
entry:
  %b = call i8* @llvm.objc.autoreleaseReturnValue(i8* %x) nounwind
  %c = call i8* @llvm.objc.unsafeClaimAutoreleasedReturnValue(i8* %y) nounwind
  ret i8* %c
}

; CHECK-LABEL: define i8* @elide_with_claimRV_wrongBB(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    call i8* @llvm.objc.autorelease(i8* %x)
; CHECK-NEXT:    br label %next
; CHECK:       next:
; CHECK-NEXT:    tail call i8* @llvm.objc.unsafeClaimAutoreleasedReturnValue(
; CHECK-NEXT:    ret i8*
define i8* @elide_with_claimRV_wrongBB(i8* %x) nounwind {
entry:
  %b = call i8* @llvm.objc.autoreleaseReturnValue(i8* %x) nounwind
  br label %next

next:
  %c = call i8* @llvm.objc.unsafeClaimAutoreleasedReturnValue(i8* %b) nounwind
  ret i8* %c
}


; CHECK-LABEL: define i8* @elide_with_claimRV_beforeAutoreleaseRV(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    tail call void @llvm.objc.release(i8* %x)
; CHECK-NEXT:    tail call i8* @llvm.objc.autoreleaseReturnValue(i8* %x)
; CHECK-NEXT:    ret i8* %x
define i8* @elide_with_claimRV_beforeAutoreleaseRV(i8* %x) nounwind {
entry:
  %b = call i8* @llvm.objc.autoreleaseReturnValue(i8* %x) nounwind
  %c = call i8* @llvm.objc.unsafeClaimAutoreleasedReturnValue(i8* %b) nounwind
  %d = call i8* @llvm.objc.autoreleaseReturnValue(i8* %c) nounwind
  ret i8* %c
}

; CHECK-LABEL: define i8* @elide_with_claimRV_afterRetain(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    ret i8* %x
define i8* @elide_with_claimRV_afterRetain(i8* %x) nounwind {
entry:
  %a = call i8* @llvm.objc.retain(i8* %x) nounwind
  %b = call i8* @llvm.objc.autoreleaseReturnValue(i8* %a) nounwind
  %c = call i8* @llvm.objc.unsafeClaimAutoreleasedReturnValue(i8* %b) nounwind
  ret i8* %c
}
