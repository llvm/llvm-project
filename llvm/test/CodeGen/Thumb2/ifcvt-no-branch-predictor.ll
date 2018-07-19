; RUN: llc < %s -mtriple=thumbv7m -mcpu=cortex-m7 | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-BP
; RUN: llc < %s -mtriple=thumbv7m -mcpu=cortex-m3 | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-NOBP

declare void @otherfn()

; CHECK-LABEL: triangle1:
; CHECK: itt ne
; CHECK: movne
; CHECK: strne
define i32 @triangle1(i32 %n, i32* %p) {
entry:
  %tobool = icmp eq i32 %n, 0
  br i1 %tobool, label %if.end, label %if.then

if.then:
  store i32 1, i32* %p, align 4
  br label %if.end

if.end:
  tail call void @otherfn()
  ret i32 0
}

; CHECK-LABEL: triangle2:
; CHECK-BP: itttt ne
; CHECK-BP: movne
; CHECK-BP: strne
; CHECK-BP: movne
; CHECK-BP: strne
; CHECK-NOBP: cbz
; CHECK-NOBP: movs
; CHECK-NOBP: str
; CHECK-NOBP: movs
; CHECK-NOBP: str
define i32 @triangle2(i32 %n, i32* %p, i32* %q) {
entry:
  %tobool = icmp eq i32 %n, 0
  br i1 %tobool, label %if.end, label %if.then

if.then:
  store i32 1, i32* %p, align 4
  store i32 2, i32* %q, align 4
  br label %if.end

if.end:
  tail call void @otherfn()
  ret i32 0
}

; CHECK-LABEL: triangle3:
; CHECK: cbz
; CHECK: movs
; CHECK: str
; CHECK: movs
; CHECK: str
; CHECK: movs
; CHECK: str
define i32 @triangle3(i32 %n, i32* %p, i32* %q, i32* %r) {
entry:
  %tobool = icmp eq i32 %n, 0
  br i1 %tobool, label %if.end, label %if.then

if.then:
  store i32 1, i32* %p, align 4
  store i32 2, i32* %q, align 4
  store i32 3, i32* %r, align 4
  br label %if.end

if.end:
  tail call void @otherfn()
  ret i32 0
}

; CHECK-LABEL: diamond1:
; CHECK: ite eq
; CHECK: ldreq
; CHECK: strne
define i32 @diamond1(i32 %n, i32* %p) {
entry:
  %tobool = icmp eq i32 %n, 0
  br i1 %tobool, label %if.else, label %if.then

if.then:
  store i32 %n, i32* %p, align 4
  br label %if.end

if.else:
  %0 = load i32, i32* %p, align 4
  br label %if.end

if.end:
  %n.addr.0 = phi i32 [ %n, %if.then ], [ %0, %if.else ]
  tail call void @otherfn()
  ret i32 %n.addr.0
}

; CHECK-LABEL: diamond2:
; CHECK-BP: cbz
; CHECK-BP: str
; CHECK-BP: str
; CHECK-BP: b
; CHECK-BP: str
; CHECK-BP: ldr
; CHECK-NOBP: ittee
; CHECK-NOBP: streq
; CHECK-NOBP: ldreq
; CHECK-NOBP: strne
; CHECK-NOBP: strne
define i32 @diamond2(i32 %n, i32 %m, i32* %p, i32* %q) {
entry:
  %tobool = icmp eq i32 %n, 0
  br i1 %tobool, label %if.else, label %if.then

if.then:
  store i32 %n, i32* %p, align 4
  %arrayidx = getelementptr inbounds i32, i32* %p, i32 2
  store i32 %n, i32* %arrayidx, align 4
  br label %if.end

if.else:
  store i32 %m, i32* %q, align 4
  %0 = load i32, i32* %p, align 4
  br label %if.end

if.end:
  %n.addr.0 = phi i32 [ %n, %if.then ], [ %0, %if.else ]
  tail call void @otherfn()
  ret i32 %n.addr.0
}

; CHECK-LABEL: diamond3:
; CHECK: cbz
; CHECK: movs
; CHECK: str
; CHECK: b
; CHECK: ldr
; CHECK: ldr
; CHECK: adds
define i32 @diamond3(i32 %n, i32* %p, i32* %q) {
entry:
  %tobool = icmp eq i32 %n, 0
  br i1 %tobool, label %if.else, label %if.then

if.then:
  store i32 1, i32* %p, align 4
  br label %if.end

if.else:
  %0 = load i32, i32* %p, align 4
  %1 = load i32, i32* %q, align 4
  %add = add nsw i32 %1, %0
  br label %if.end

if.end:
  %n.addr.0 = phi i32 [ %n, %if.then ], [ %add, %if.else ]
  tail call void @otherfn()
  ret i32 %n.addr.0
}
