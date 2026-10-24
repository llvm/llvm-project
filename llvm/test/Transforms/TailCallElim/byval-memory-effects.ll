; RUN: opt -S -passes=tailcallelim -verify-each %s | FileCheck %s --check-prefix=TRE
; RUN: opt -S -passes='function(tailcallelim),cgscc(inline)' -verify-each %s | FileCheck %s --check-prefix=INLINE
; RUN: opt -S -passes='forceattrs,cgscc(function-attrs),function(tailcallelim),cgscc(inline)' -force-remove-attribute=f:memory -verify-each %s | FileCheck %s --check-prefix=INLINE
;
; Tail recursion elimination copies new byval values into the function's
; incoming arguments. Its memory effects must account for those accesses,
; including when the original effects were inferred by function-attrs.
; Otherwise the inliner can elide the byval copy and overwrite the caller's
; object. The incoming object in main must retain its original second field.

target datalayout = "e-p:64:64-i64:64"

%T = type { i64, i32 }

define internal i64 @f(ptr readonly byval(%T) align 8 %p) memory(argmem: read) {
; TRE-LABEL: define internal i64 @f(
; TRE-SAME: ptr byval(%T) align 8 %p) #[[READWRITE:[0-9]+]] {
; TRE: call void @llvm.memcpy.p0.p0.i64(ptr align 8 %p,
entry:
  %local = alloca %T, align 8
  %v = load %T, ptr %p, align 8
  store %T %v, ptr %local, align 8
  %n.addr = getelementptr inbounds %T, ptr %local, i32 0, i32 1
  %n = load i32, ptr %n.addr, align 8
  %done = icmp eq i32 %n, 0
  br i1 %done, label %exit, label %recurse

exit:
  ret i64 0

recurse:
  store i32 0, ptr %n.addr, align 8
  %r = call i64 @f(ptr byval(%T) align 8 %local)
  ret i64 %r
}

define i32 @main() {
; INLINE-LABEL: define i32 @main(
; INLINE: [[COPY:%s[0-9]+]] = alloca %T, align 8
; INLINE: %s = alloca %T, align 8
; INLINE: call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[COPY]], ptr align 8 %s, i64 16, i1 false)
entry:
  %s = alloca %T, align 8
  store %T { i64 1, i32 5 }, ptr %s, align 8
  %r = call i64 @f(ptr byval(%T) align 8 %s)
  %n.addr = getelementptr inbounds %T, ptr %s, i32 0, i32 1
  %n = load volatile i32, ptr %n.addr, align 8
  ret i32 %n
}

; Introducing argument-memory accesses must preserve effects for other memory
; locations, even when the original function did not access argument memory.
define void @preserve_effects(ptr byval(%T) align 8 %p, i1 %done) memory(inaccessiblemem: write) {
; TRE-LABEL: define void @preserve_effects(
; TRE-SAME: #[[PRESERVED:[0-9]+]] {
; TRE: call void @llvm.memcpy.p0.p0.i64(ptr align 8 %p,
entry:
  %local = alloca %T, align 8
  store %T zeroinitializer, ptr %local, align 8
  br i1 %done, label %exit, label %recurse

exit:
  ret void

recurse:
  call void @preserve_effects(ptr byval(%T) align 8 %local, i1 true)
  ret void
}

; Unrestricted memory effects already allow the copies.
define void @unrestricted(ptr byval(%T) align 8 %p, i1 %done) {
; TRE-LABEL: define void @unrestricted(
; TRE-SAME: i1 %done) {
entry:
  %local = alloca %T, align 8
  store %T zeroinitializer, ptr %local, align 8
  br i1 %done, label %exit, label %recurse

exit:
  ret void

recurse:
  call void @unrestricted(ptr byval(%T) align 8 %local, i1 true)
  ret void
}

; Tail recursion without byval arguments must not gain additional memory effects.
define i32 @no_byval(ptr %p, i1 %done) memory(argmem: read) {
; TRE-LABEL: define i32 @no_byval(
; TRE-SAME: #[[READONLY:[0-9]+]] {
entry:
  br i1 %done, label %exit, label %recurse

exit:
  %v = load i32, ptr %p, align 4
  ret i32 %v

recurse:
  %r = call i32 @no_byval(ptr %p, i1 true)
  ret i32 %r
}

; TRE: attributes #[[READWRITE]] = { memory(argmem: readwrite) }
; TRE: attributes #[[PRESERVED]] = { memory(argmem: readwrite, inaccessiblemem: write) }
; TRE: attributes #[[READONLY]] = { memory(argmem: read) }
