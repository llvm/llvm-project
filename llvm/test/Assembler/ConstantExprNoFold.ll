; This test checks to make sure that constant exprs don't fold in some simple
; situations

; RUN: llvm-as < %s | llvm-dis | FileCheck %s
; RUN: verify-uselistorder %s

; Even give it a datalayout, to tempt folding as much as possible.
target datalayout = "p:32:32"

@A = global i64 0
@B = global i64 0

; Don't fold this. @A might really be allocated next to @B, in which case the
; icmp should return true. It's not valid to *dereference* in @B from a pointer
; based on @A, but icmp isn't a dereference.

; CHECK: @C = global i1 icmp eq (i64* getelementptr inbounds (i64, i64* @A, i64 1), i64* @B)
@C = global i1 icmp eq (i64* getelementptr inbounds (i64, i64* @A, i64 1), i64* @B)

; Don't fold this completely away either. In theory this could be simplified
; to only use a gep on one side of the icmp though.

; CHECK: @D = global i1 icmp eq (i64* getelementptr inbounds (i64, i64* @A, i64 1), i64* getelementptr inbounds (i64, i64* @B, i64 2))
@D = global i1 icmp eq (i64* getelementptr inbounds (i64, i64* @A, i64 1), i64* getelementptr inbounds (i64, i64* @B, i64 2))

; CHECK: @E = global i64 addrspace(1)* addrspacecast (i64* @A to i64 addrspace(1)*)
@E = global i64 addrspace(1)* addrspacecast(i64* @A to i64 addrspace(1)*)

; Don't add an inbounds on @weak.gep, since @weak may be null.
; CHECK: @weak.gep = global i32* getelementptr (i32, i32* @weak, i32 1)
@weak.gep = global i32* getelementptr (i32, i32* @weak, i32 1)
@weak = extern_weak global i32

; An object with weak linkage cannot have it's identity determined at compile time.
; CHECK: @F = global i1 icmp eq (i32* @weakany, i32* @glob)
@F = global i1 icmp eq (i32* @weakany, i32* @glob)
@weakany = weak global i32 0

; Empty globals might end up anywhere, even on top of another global.
; CHECK: @empty.cmp = global i1 icmp eq ([0 x i8]* @empty.1, [0 x i8]* @empty.2)
@empty.1 = external global [0 x i8], align 1
@empty.2 = external global [0 x i8], align 1
@empty.cmp = global i1 icmp eq ([0 x i8]* @empty.1, [0 x i8]* @empty.2)

@addrspace3 = internal addrspace(3) global i32 undef

; CHECK: @no.fold.addrspace.icmp.eq.gv.null = global i1 icmp eq (i32 addrspace(3)* @addrspace3, i32 addrspace(3)* null)
; CHECK: @no.fold.addrspace.icmp.eq.null.gv = global i1 icmp eq (i32 addrspace(3)* @addrspace3, i32 addrspace(3)* null)
; CHECK: @no.fold.addrspace.icmp.ne.gv.null = global i1 icmp ne (i32 addrspace(3)* @addrspace3, i32 addrspace(3)* null)
; CHECK: @no.fold.addrspace.icmp.ne.null.gv = global i1 icmp ne (i32 addrspace(3)* @addrspace3, i32 addrspace(3)* null)
@no.fold.addrspace.icmp.eq.gv.null = global i1 icmp eq (i32 addrspace(3)* @addrspace3, i32 addrspace(3)* null)
@no.fold.addrspace.icmp.eq.null.gv = global i1 icmp eq (i32 addrspace(3)* null, i32 addrspace(3)* @addrspace3)
@no.fold.addrspace.icmp.ne.gv.null = global i1 icmp ne (i32 addrspace(3)* @addrspace3, i32 addrspace(3)* null)
@no.fold.addrspace.icmp.ne.null.gv = global i1 icmp ne (i32 addrspace(3)* null, i32 addrspace(3)* @addrspace3)

; Don't add an inbounds on @glob.a3, since it's not inbounds.
; CHECK: @glob.a3 = alias i32, getelementptr (i32, i32* @glob.a2, i32 1)
@glob = global i32 0
@glob.a3 = alias i32, getelementptr (i32, i32* @glob.a2, i32 1)
@glob.a2 = alias i32, getelementptr (i32, i32* @glob.a1, i32 1)
@glob.a1 = alias i32, i32* @glob
