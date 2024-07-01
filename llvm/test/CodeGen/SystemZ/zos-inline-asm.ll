; RUN: llc -mtriple s390x-ibm-zos < %s | FileCheck %s
; Source to generate .ll file
;
; void f1() {
;   int a, b;
;   __asm(" lhi %0,5\n"
;         : "={r1}"(a)
;         :
;         :);
;
;   __asm(" lgr %0,%1\n"
;         : "={r1}"(a)
;         : "{r2}"(b)
;         :);
; }
;
; void f2() {
;   int a, m_b;
;   __asm(" stg %1,%0\n"
;         : "=m"(m_b)
;         : "{r1}"(a)
;         :);
; }
;
; void f3() {
;   int r15, r1;
;
;   __asm(" svc 109\n"
;         : "={r15}"(r15)
;         : "{r1}"(r1), "{r15}"(25)
;         :);
; }
;
; void f4() {
;   ptr parm;
;   long long rc, reason;
;   char *code;
;
;   __asm(" pc 0(%3)"
;         : "={r0}"(reason), "+{r1}"(parm), "={r15}"(rc)
;         : "r"(code)
;         :);
; }
;
; void f5() {
;
;   int a;
;   int b;
;   int c;
;
;   __asm(" lhi %0,10\n"
;         " ar %0,%0\n"
;         : "=&r"(a)
;         :
;         :);
;
;   __asm(" lhi %0,10\n"
;         " ar %0,%0\n"
;         : "={&r2}"(b)
;         :
;         :);
;
;   __asm(" lhi %0,10\n"
;         " ar %0,%0\n"
;         : "={&r2}"(c)
;         :
;         :);
; }
;
; void f7() {
;   int a, b, res;
;
;   a = 2147483640;
;   b = 10;
;
;   __asm(" alr %0,%1\n"
;         " jo *-4\n"
;         :"=r"(res)
;         :"r"(a), "r"(b)
;         :);
; }
;
; int f8() {
;
;   int a, b, res;
;   a = b = res = -1;
;
;   __asm(" lhi 1,5\n"
;         :
;         :
;         : "r1");
;
;   __asm(" lgr 2,1\n"
;         :
;         :
;         : "r2");
;
;   __asm(" stg 2,%0\n"
;         :
;         : "r"(res)
;         :);
;
;  return res;
; }

define hidden void @f1() {
; CHECK-LABEL: f1:
; CHECK: *APP
; CHECK-NEXT: lhi 1, 5
; CHECK: *NO_APP
; CHECK: *APP
; CHECK-NEXT: lgr 1, 2
; CHECK: *NO_APP
entry:
  %a = alloca i32, align 4
  %b = alloca i32, align 4
  %0 = call i32 asm " lhi $0,5\0A", "={r1}"()
  store i32 %0, ptr %a, align 4
  %1 = load i32, ptr %b, align 4
  %2 = call i32 asm " lgr $0,$1\0A", "={r1},{r2}"(i32 %1)
  store i32 %2, ptr %a, align 4
  ret void
}

define hidden void @f2() {
; CHECK-LABEL: f2:
; CHECK: *APP
; CHECK-NEXT: stg 1, {{.*}}(4)
; CHECK: *NO_APP
entry:
  %a = alloca i32, align 4
  %m_b = alloca i32, align 4
  %0 = load i32, ptr %a, align 4
  call void asm " stg $1,$0\0A", "=*m,{r1}"(ptr elementtype(i32) %m_b, i32 %0)
  ret void
}

define hidden void @f3() {
; CHECK-LABEL: f3:
; CHECK: l 1, {{.*}}(4)
; CHECK: lhi 15, 25
; CHECK: *APP
; CHECK-NEXT: svc 109
; CHECK: *NO_APP
entry:
  %r15 = alloca i32, align 4
  %r1 = alloca i32, align 4
  %0 = load i32, ptr %r1, align 4
  %1 = call i32 asm " svc 109\0A", "={r15},{r1},{r15}"(i32 %0, i32 25)
  store i32 %1, ptr %r15, align 4
  ret void
}

define hidden void @f4() {
; CHECK-LABEL: f4:
; CHECK: *APP
; CHECK-NEXT: pc 0
; CHECK: *NO_APP
; CHECK: stg 0, {{.*}}(4)
; CHECK-NEXT: stg 1, {{.*}}(4)
; CHECK-NEXT: stg 15, {{.*}}(4)
entry:
  %parm = alloca ptr, align 8
  %rc = alloca i64, align 8
  %reason = alloca i64, align 8
  %code = alloca ptr, align 8
  %0 = load ptr, ptr %parm, align 8
  %1 = load ptr, ptr %code, align 8
  %2 = call { i64, ptr, i64 } asm " pc 0($3)", "={r0},={r1},={r15},r,1"(ptr %1, ptr %0)
  %asmresult = extractvalue { i64, ptr, i64 } %2, 0
  %asmresult1 = extractvalue { i64, ptr, i64 } %2, 1
  %asmresult2 = extractvalue { i64, ptr, i64 } %2, 2
  store i64 %asmresult, ptr %reason, align 8
  store ptr %asmresult1, ptr %parm, align 8
  store i64 %asmresult2, ptr %rc, align 8
  ret void
}

define hidden void @f5() {
; CHECK-LABEL: f5:
; CHECK: *APP
; CHECK-NEXT: lhi {{[0-9]}}, 10
; CHECK-NEXT: ar {{[0-9]}}, {{[0-9]}}
; CHECK: *NO_APP
; CHECK: *APP
; CHECK-NEXT: lhi 2, 10
; CHECK-NEXT: ar 2, 2
; CHECK: *NO_APP
; CHECK: *APP
; CHECK-NEXT: lhi 2, 10
; CHECK-NEXT: ar 2, 2
; CHECK: *NO_APP
entry:
  %a = alloca i32, align 4
  %b = alloca i32, align 4
  %c = alloca i32, align 4
  %0 = call i32 asm " lhi $0,10\0A ar $0,$0\0A", "=&r"()
  store i32 %0, ptr %a, align 4
  %1 = call i32 asm " lhi $0,10\0A ar $0,$0\0A", "=&{r2}"()
  store i32 %1, ptr %b, align 4
  %2 = call i32 asm " lhi $0,10\0A ar $0,$0\0A", "=&{r2}"()
  store i32 %2, ptr %c, align 4
  ret void
}

define hidden void @f7() {
; CHECK-LABEL: f7:
; CHECK: *APP
; CHECK-NEXT: alr {{[0-9]}}, {{[0-9]}}
; CHECK-NEXT: {{.*}}:
; CHECK-NEXT: jo {{.*}}-4
; CHECK: *NO_APP
entry:
  %a = alloca i32, align 4
  %b = alloca i32, align 4
  %res = alloca i32, align 4
  store i32 2147483640, ptr %a, align 4
  store i32 10, ptr %b, align 4
  %0 = load i32, ptr %a, align 4
  %1 = load i32, ptr %b, align 4
  %2 = call i32 asm " alr $0,$1\0A jo *-4\0A", "=r,r,r"(i32 %0, i32 %1)
  store i32 %2, ptr %res, align 4
  ret void
}

define hidden signext i32 @f8() {
; CHECK-LABEL: f8:
; CHECK: *APP
; CHECK-NEXT: lhi 1, 5
; CHECK: *NO_APP
; CHECK: *APP
; CHECK-NEXT: lgr 2, 1
; CHECK: *NO_APP
; CHECK: *APP
; CHECK-NEXT: stg 2, {{.*}}(4)
; CHECK: *NO_APP
; CHECK: lgf 3, {{.*}}(4)
entry:
  %a = alloca i32, align 4
  %b = alloca i32, align 4
  %res = alloca i32, align 4
  store i32 -1, ptr %res, align 4
  store i32 -1, ptr %b, align 4
  store i32 -1, ptr %a, align 4
  call void asm sideeffect " lhi 1,5\0A", "~{r1}"()
  call void asm sideeffect " lgr 2,1\0A", "~{r2}"()
  call void asm " stg 2,$0\0A", "=*m"(ptr elementtype(i32) %res)
  %0 = load i32, ptr %res, align 4
  ret i32 %0
}
