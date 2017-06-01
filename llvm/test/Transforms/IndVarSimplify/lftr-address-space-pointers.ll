; RUN: opt -S -indvars -o - %s | FileCheck %s
target datalayout = "e-p:32:32:32-p1:64:64:64-p2:8:8:8-p3:16:16:16-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:32-n8:16:32:64"

; Derived from ptriv in lftr-reuse.ll
define void @ptriv_as2(i8 addrspace(2)* %base, i32 %n) nounwind {
; CHECK-LABEL: @ptriv_as2(
entry:
  %idx.trunc = trunc i32 %n to i8
  %add.ptr = getelementptr inbounds i8, i8 addrspace(2)* %base, i8 %idx.trunc
  %cmp1 = icmp ult i8 addrspace(2)* %base, %add.ptr
  br i1 %cmp1, label %for.body, label %for.end

; Make sure the added GEP has the right index type
; CHECK: %lftr.limit = getelementptr i8, i8 addrspace(2)* %base, i8 %idx.trunc

; CHECK: for.body:
; CHECK: phi i8 addrspace(2)*
; CHECK-NOT: phi
; CHECK-NOT: add{{^rspace}}
; CHECK: icmp ne i8 addrspace(2)*
; CHECK: br i1
for.body:
  %p.02 = phi i8 addrspace(2)* [ %base, %entry ], [ %incdec.ptr, %for.body ]
  ; cruft to make the IV useful
  %sub.ptr.lhs.cast = ptrtoint i8 addrspace(2)* %p.02 to i8
  %sub.ptr.rhs.cast = ptrtoint i8 addrspace(2)* %base to i8
  %sub.ptr.sub = sub i8 %sub.ptr.lhs.cast, %sub.ptr.rhs.cast
  store i8 %sub.ptr.sub, i8 addrspace(2)* %p.02
  %incdec.ptr = getelementptr inbounds i8, i8 addrspace(2)* %p.02, i32 1
  %cmp = icmp ult i8 addrspace(2)* %incdec.ptr, %add.ptr
  br i1 %cmp, label %for.body, label %for.end

for.end:
  ret void
}

define void @ptriv_as3(i8 addrspace(3)* %base, i32 %n) nounwind {
; CHECK-LABEL: @ptriv_as3(
entry:
  %idx.trunc = trunc i32 %n to i16
  %add.ptr = getelementptr inbounds i8, i8 addrspace(3)* %base, i16 %idx.trunc
  %cmp1 = icmp ult i8 addrspace(3)* %base, %add.ptr
  br i1 %cmp1, label %for.body, label %for.end

; Make sure the added GEP has the right index type
; CHECK: %lftr.limit = getelementptr i8, i8 addrspace(3)* %base, i16 %idx.trunc

; CHECK: for.body:
; CHECK: phi i8 addrspace(3)*
; CHECK-NOT: phi
; CHECK-NOT: add{{^rspace}}
; CHECK: icmp ne i8 addrspace(3)*
; CHECK: br i1
for.body:
  %p.02 = phi i8 addrspace(3)* [ %base, %entry ], [ %incdec.ptr, %for.body ]
  ; cruft to make the IV useful
  %sub.ptr.lhs.cast = ptrtoint i8 addrspace(3)* %p.02 to i16
  %sub.ptr.rhs.cast = ptrtoint i8 addrspace(3)* %base to i16
  %sub.ptr.sub = sub i16 %sub.ptr.lhs.cast, %sub.ptr.rhs.cast
  %conv = trunc i16 %sub.ptr.sub to i8
  store i8 %conv, i8 addrspace(3)* %p.02
  %incdec.ptr = getelementptr inbounds i8, i8 addrspace(3)* %p.02, i32 1
  %cmp = icmp ult i8 addrspace(3)* %incdec.ptr, %add.ptr
  br i1 %cmp, label %for.body, label %for.end

for.end:
  ret void
}

