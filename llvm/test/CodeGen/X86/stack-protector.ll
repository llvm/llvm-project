; RUN: llc -mtriple=i386-pc-linux-gnu < %s -o - | FileCheck --check-prefix=LINUX-I386 %s
; RUN: llc -mtriple=x86_64-pc-linux-gnu < %s -o - | FileCheck --check-prefix=LINUX-X64 %s
; RUN: llc -code-model=kernel -mtriple=x86_64-pc-linux-gnu < %s -o - | FileCheck --check-prefix=LINUX-KERNEL-X64 %s
; RUN: llc -mtriple=x86_64-apple-darwin < %s -o - | FileCheck --check-prefix=DARWIN-X64 %s
; RUN: llc -mtriple=amd64-pc-openbsd < %s -o - | FileCheck --check-prefix=OPENBSD-AMD64 %s
; RUN: llc -mtriple=i386-pc-windows-msvc < %s -o - | FileCheck -check-prefix=MSVC-I386 %s
; RUN: llc -mtriple=x86_64-w64-mingw32 < %s -o - | FileCheck --check-prefix=MINGW-X64 %s
; RUN: llc -mtriple=x86_64-pc-linux-gnu < %s -o - | FileCheck --check-prefix=IGNORE_INTRIN %s

%struct.foo = type { [16 x i8] }
%struct.foo.0 = type { [4 x i8] }
%struct.pair = type { i32, i32 }
%struct.nest = type { %struct.pair, %struct.pair }
%struct.vec = type { <4 x i32> }
%class.A = type { [2 x i8] }
%struct.deep = type { %union.anon }
%union.anon = type { %struct.anon }
%struct.anon = type { %struct.anon.0 }
%struct.anon.0 = type { %union.anon.1 }
%union.anon.1 = type { [2 x i8] }
%struct.small = type { i8 }
%struct.small_char = type { i32, [5 x i8] }

@.str = private unnamed_addr constant [4 x i8] c"%s\0A\00", align 1

; test1a: array of [16 x i8] 
;         no ssp attribute
; Requires no protector.
define void @test1a(ptr %a) {
entry:
; LINUX-I386-LABEL: test1a:
; LINUX-I386-NOT: calll __stack_chk_fail
; LINUX-I386: .cfi_endproc

; LINUX-X64-LABEL: test1a:
; LINUX-X64-NOT: callq __stack_chk_fail
; LINUX-X64: .cfi_endproc

; LINUX-KERNEL-X64-LABEL: test1a:
; LINUX-KERNEL-X64-NOT: callq __stack_chk_fail
; LINUX-KERNEL-X64: .cfi_endproc

; DARWIN-X64-LABEL: test1a:
; DARWIN-X64-NOT: callq ___stack_chk_fail
; DARWIN-X64: .cfi_endproc

; MSVC-I386-LABEL: test1a:
; MSVC-I386-NOT: calll  @__security_check_cookie@4
; MSVC-I386: retl

; MINGW-X64-LABEL: test1a:
; MINGW-X64-NOT: callq __stack_chk_fail
; MINGW-X64: .seh_endproc

  %a.addr = alloca ptr, align 8
  %buf = alloca [16 x i8], align 16
  store ptr %a, ptr %a.addr, align 8
  %0 = load ptr, ptr %a.addr, align 8
  %call = call ptr @strcpy(ptr %buf, ptr %0)
  %call2 = call i32 (ptr, ...) @printf(ptr @.str, ptr %buf)
  ret void
}

; test1b: array of [16 x i8] 
;         ssp attribute
; Requires protector.
; Function Attrs: ssp
define void @test1b(ptr %a) #0 {
entry:
; LINUX-I386-LABEL: test1b:
; LINUX-I386: mov{{l|q}} %gs:
; LINUX-I386: calll __stack_chk_fail

; LINUX-X64-LABEL: test1b:
; LINUX-X64: mov{{l|q}} %fs:
; LINUX-X64: callq __stack_chk_fail

; LINUX-KERNEL-X64-LABEL: test1b:
; LINUX-KERNEL-X64: mov{{l|q}} %gs:
; LINUX-KERNEL-X64: callq __stack_chk_fail

; DARWIN-X64-LABEL: test1b:
; DARWIN-X64: mov{{l|q}} ___stack_chk_guard
; DARWIN-X64: callq ___stack_chk_fail

; OPENBSD-AMD64-LABEL: test1b:
; OPENBSD-AMD64: movq __guard_local(%rip)
; OPENBSD-AMD64: callq __stack_smash_handler

; MSVC-I386-LABEL: test1b:
; MSVC-I386: movl ___security_cookie,
; MSVC-I386: calll @__security_check_cookie@4

; MINGW-X64-LABEL: test1b:
; MINGW-X64: mov{{l|q}} .refptr.__stack_chk_guard
; MINGW-X64: callq __stack_chk_fail

  %a.addr = alloca ptr, align 8
  %buf = alloca [16 x i8], align 16
  store ptr %a, ptr %a.addr, align 8
  %0 = load ptr, ptr %a.addr, align 8
  %call = call ptr @strcpy(ptr %buf, ptr %0)
  %call2 = call i32 (ptr, ...) @printf(ptr @.str, ptr %buf)
  ret void
}

; test1c: array of [16 x i8] 
;         sspstrong attribute
; Requires protector.
; Function Attrs: sspstrong 
define void @test1c(ptr %a) #1 {
entry:
; LINUX-I386-LABEL: test1c:
; LINUX-I386: mov{{l|q}} %gs:
; LINUX-I386: calll __stack_chk_fail

; LINUX-X64-LABEL: test1c:
; LINUX-X64: mov{{l|q}} %fs:
; LINUX-X64: callq __stack_chk_fail

; LINUX-KERNEL-X64-LABEL: test1c:
; LINUX-KERNEL-X64: mov{{l|q}} %gs:
; LINUX-KERNEL-X64: callq __stack_chk_fail

; DARWIN-X64-LABEL: test1c:
; DARWIN-X64: mov{{l|q}} ___stack_chk_guard
; DARWIN-X64: callq ___stack_chk_fail

; MSVC-I386-LABEL: test1c:
; MSVC-I386: movl ___security_cookie,
; MSVC-I386: calll @__security_check_cookie@4

; MINGW-X64-LABEL: test1c:
; MINGW-X64: mov{{l|q}} .refptr.__stack_chk_guard
; MINGW-X64: callq __stack_chk_fail

  %a.addr = alloca ptr, align 8
  %buf = alloca [16 x i8], align 16
  store ptr %a, ptr %a.addr, align 8
  %0 = load ptr, ptr %a.addr, align 8
  %call = call ptr @strcpy(ptr %buf, ptr %0)
  %call2 = call i32 (ptr, ...) @printf(ptr @.str, ptr %buf)
  ret void
}

; test1d: array of [16 x i8] 
;         sspreq attribute
; Requires protector.
; Function Attrs: sspreq 
define void @test1d(ptr %a) #2 {
entry:
; LINUX-I386-LABEL: test1d:
; LINUX-I386: mov{{l|q}} %gs:
; LINUX-I386: calll __stack_chk_fail

; LINUX-X64-LABEL: test1d:
; LINUX-X64: mov{{l|q}} %fs:
; LINUX-X64: callq __stack_chk_fail

; LINUX-KERNEL-X64-LABEL: test1d:
; LINUX-KERNEL-X64: mov{{l|q}} %gs:
; LINUX-KERNEL-X64: callq __stack_chk_fail

; DARWIN-X64-LABEL: test1d:
; DARWIN-X64: mov{{l|q}} ___stack_chk_guard
; DARWIN-X64: callq ___stack_chk_fail

; MSVC-I386-LABEL: test1d:
; MSVC-I386: movl ___security_cookie,
; MSVC-I386: calll @__security_check_cookie@4

; MINGW-X64-LABEL: test1d:
; MINGW-X64: mov{{l|q}} .refptr.__stack_chk_guard
; MINGW-X64: callq __stack_chk_fail

  %a.addr = alloca ptr, align 8
  %buf = alloca [16 x i8], align 16
  store ptr %a, ptr %a.addr, align 8
  %0 = load ptr, ptr %a.addr, align 8
  %call = call ptr @strcpy(ptr %buf, ptr %0)
  %call2 = call i32 (ptr, ...) @printf(ptr @.str, ptr %buf)
  ret void
}

; test2a: struct { [16 x i8] }
;         no ssp attribute
; Requires no protector.
define void @test2a(ptr %a) {
entry:
; LINUX-I386-LABEL: test2a:
; LINUX-I386-NOT: calll __stack_chk_fail
; LINUX-I386: .cfi_endproc

; LINUX-X64-LABEL: test2a:
; LINUX-X64-NOT: callq __stack_chk_fail
; LINUX-X64: .cfi_endproc

; LINUX-KERNEL-X64-LABEL: test2a:
; LINUX-KERNEL-X64-NOT: callq __stack_chk_fail
; LINUX-KERNEL-X64: .cfi_endproc

; DARWIN-X64-LABEL: test2a:
; DARWIN-X64-NOT: callq ___stack_chk_fail
; DARWIN-X64: .cfi_endproc

; MSVC-I386-LABEL: test2a:
; MSVC-I386-NOT: calll @__security_check_cookie@4
; MSVC-I386: retl

; MINGW-X64-LABEL: test2a:
; MINGW-X64-NOT: callq __stack_chk_fail
; MINGW-X64: .seh_endproc

  %a.addr = alloca ptr, align 8
  %b = alloca %struct.foo, align 1
  store ptr %a, ptr %a.addr, align 8
  %0 = load ptr, ptr %a.addr, align 8
  %call = call ptr @strcpy(ptr %b, ptr %0)
  %call3 = call i32 (ptr, ...) @printf(ptr @.str, ptr %b)
  ret void
}

; test2b: struct { [16 x i8] }
;          ssp attribute
; Requires protector.
; Function Attrs: ssp
define void @test2b(ptr %a) #0 {
entry:
; LINUX-I386-LABEL: test2b:
; LINUX-I386: mov{{l|q}} %gs:
; LINUX-I386: calll __stack_chk_fail

; LINUX-X64-LABEL: test2b:
; LINUX-X64: mov{{l|q}} %fs:
; LINUX-X64: callq __stack_chk_fail

; LINUX-KERNEL-X64-LABEL: test2b:
; LINUX-KERNEL-X64: mov{{l|q}} %gs:
; LINUX-KERNEL-X64: callq __stack_chk_fail

; DARWIN-X64-LABEL: test2b:
; DARWIN-X64: mov{{l|q}} ___stack_chk_guard
; DARWIN-X64: callq ___stack_chk_fail

; MINGW-X64-LABEL: test2b:
; MINGW-X64: mov{{l|q}} .refptr.__stack_chk_guard
; MINGW-X64: callq __stack_chk_fail

  %a.addr = alloca ptr, align 8
  %b = alloca %struct.foo, align 1
  store ptr %a, ptr %a.addr, align 8
  %0 = load ptr, ptr %a.addr, align 8
  %call = call ptr @strcpy(ptr %b, ptr %0)
  %call3 = call i32 (ptr, ...) @printf(ptr @.str, ptr %b)
  ret void
}

; test2c: struct { [16 x i8] }
;          sspstrong attribute
; Requires protector.
; Function Attrs: sspstrong 
define void @test2c(ptr %a) #1 {
entry:
; LINUX-I386-LABEL: test2c:
; LINUX-I386: mov{{l|q}} %gs:
; LINUX-I386: calll __stack_chk_fail

; LINUX-X64-LABEL: test2c:
; LINUX-X64: mov{{l|q}} %fs:
; LINUX-X64: callq __stack_chk_fail

; LINUX-KERNEL-X64-LABEL: test2c:
; LINUX-KERNEL-X64: mov{{l|q}} %gs:
; LINUX-KERNEL-X64: callq __stack_chk_fail

; DARWIN-X64-LABEL: test2c:
; DARWIN-X64: mov{{l|q}} ___stack_chk_guard
; DARWIN-X64: callq ___stack_chk_fail

; MSVC-I386-LABEL: test2c:
; MSVC-I386: movl ___security_cookie,
; MSVC-I386: calll @__security_check_cookie@4

; MINGW-X64-LABEL: test2c:
; MINGW-X64: mov{{l|q}} .refptr.__stack_chk_guard
; MINGW-X64: callq __stack_chk_fail

  %a.addr = alloca ptr, align 8
  %b = alloca %struct.foo, align 1
  store ptr %a, ptr %a.addr, align 8
  %0 = load ptr, ptr %a.addr, align 8
  %call = call ptr @strcpy(ptr %b, ptr %0)
  %call3 = call i32 (ptr, ...) @printf(ptr @.str, ptr %b)
  ret void
}

; test2d: struct { [16 x i8] }
;          sspreq attribute
; Requires protector.
; Function Attrs: sspreq 
define void @test2d(ptr %a) #2 {
entry:
; LINUX-I386-LABEL: test2d:
; LINUX-I386: mov{{l|q}} %gs:
; LINUX-I386: calll __stack_chk_fail

; LINUX-X64-LABEL: test2d:
; LINUX-X64: mov{{l|q}} %fs:
; LINUX-X64: callq __stack_chk_fail

; LINUX-KERNEL-X64-LABEL: test2d:
; LINUX-KERNEL-X64: mov{{l|q}} %gs:
; LINUX-KERNEL-X64: callq __stack_chk_fail

; DARWIN-X64-LABEL: test2d:
; DARWIN-X64: mov{{l|q}} ___stack_chk_guard
; DARWIN-X64: callq ___stack_chk_fail

; MSVC-I386-LABEL: test2d:
; MSVC-I386: movl ___security_cookie,
; MSVC-I386: calll @__security_check_cookie@4

; MINGW-X64-LABEL: test2d:
; MINGW-X64: mov{{l|q}} .refptr.__stack_chk_guard
; MINGW-X64: callq __stack_chk_fail

  %a.addr = alloca ptr, align 8
  %b = alloca %struct.foo, align 1
  store ptr %a, ptr %a.addr, align 8
  %0 = load ptr, ptr %a.addr, align 8
  %call = call ptr @strcpy(ptr %b, ptr %0)
  %call3 = call i32 (ptr, ...) @printf(ptr @.str, ptr %b)
  ret void
}

; test3a:  array of [4 x i8]
;          no ssp attribute
; Requires no protector.
define void @test3a(ptr %a) {
entry:
; LINUX-I386-LABEL: test3a:
; LINUX-I386-NOT: calll __stack_chk_fail
; LINUX-I386: .cfi_endproc

; LINUX-X64-LABEL: test3a:
; LINUX-X64-NOT: callq __stack_chk_fail
; LINUX-X64: .cfi_endproc

; LINUX-KERNEL-X64-LABEL: test3a:
; LINUX-KERNEL-X64-NOT: callq __stack_chk_fail
; LINUX-KERNEL-X64: .cfi_endproc

; DARWIN-X64-LABEL: test3a:
; DARWIN-X64-NOT: callq ___stack_chk_fail
; DARWIN-X64: .cfi_endproc

; MSVC-I386-LABEL: test3a:
; MSVC-I386-NOT: calll @__security_check_cookie@4
; MSVC-I386: retl

; MINGW-X64-LABEL: test3a:
; MINGW-X64-NOT: callq __stack_chk_fail
; MINGW-X64: .seh_endproc

  %a.addr = alloca ptr, align 8
  %buf = alloca [4 x i8], align 1
  store ptr %a, ptr %a.addr, align 8
  %0 = load ptr, ptr %a.addr, align 8
  %call = call ptr @strcpy(ptr %buf, ptr %0)
  %call2 = call i32 (ptr, ...) @printf(ptr @.str, ptr %buf)
  ret void
}

; test3b:  array [4 x i8]
;          ssp attribute
; Requires no protector.
; Function Attrs: ssp
define void @test3b(ptr %a) #0 {
entry:
; LINUX-I386-LABEL: test3b:
; LINUX-I386-NOT: calll __stack_chk_fail
; LINUX-I386: .cfi_endproc

; LINUX-X64-LABEL: test3b:
; LINUX-X64-NOT: callq __stack_chk_fail
; LINUX-X64: .cfi_endproc

; LINUX-KERNEL-X64-LABEL: test3b:
; LINUX-KERNEL-X64-NOT: callq __stack_chk_fail
; LINUX-KERNEL-X64: .cfi_endproc

; DARWIN-X64-LABEL: test3b:
; DARWIN-X64-NOT: callq ___stack_chk_fail
; DARWIN-X64: .cfi_endproc

; MSVC-I386-LABEL: test3b:
; MSVC-I386-NOT: calll @__security_check_cookie@4
; MSVC-I386: retl

; MINGW-X64-LABEL: test3b:
; MINGW-X64-NOT: callq __stack_chk_fail
; MINGW-X64: .seh_endproc

  %a.addr = alloca ptr, align 8
  %buf = alloca [4 x i8], align 1
  store ptr %a, ptr %a.addr, align 8
  %0 = load ptr, ptr %a.addr, align 8
  %call = call ptr @strcpy(ptr %buf, ptr %0)
  %call2 = call i32 (ptr, ...) @printf(ptr @.str, ptr %buf)
  ret void
}

; test3c:  array of [4 x i8]
;          sspstrong attribute
; Requires protector.
; Function Attrs: sspstrong 
define void @test3c(ptr %a) #1 {
entry:
; LINUX-I386-LABEL: test3c:
; LINUX-I386: mov{{l|q}} %gs:
; LINUX-I386: calll __stack_chk_fail

; LINUX-X64-LABEL: test3c:
; LINUX-X64: mov{{l|q}} %fs:
; LINUX-X64: callq __stack_chk_fail

; LINUX-KERNEL-X64-LABEL: test3c:
; LINUX-KERNEL-X64: mov{{l|q}} %gs:
; LINUX-KERNEL-X64: callq __stack_chk_fail

; DARWIN-X64-LABEL: test3c:
; DARWIN-X64: mov{{l|q}} ___stack_chk_guard
; DARWIN-X64: callq ___stack_chk_fail

; MSVC-I386-LABEL: test3c:
; MSVC-I386: movl ___security_cookie,
; MSVC-I386: calll @__security_check_cookie@4

; MINGW-X64-LABEL: test3c:
; MINGW-X64: mov{{l|q}} .refptr.__stack_chk_guard
; MINGW-X64: callq __stack_chk_fail

  %a.addr = alloca ptr, align 8
  %buf = alloca [4 x i8], align 1
  store ptr %a, ptr %a.addr, align 8
  %0 = load ptr, ptr %a.addr, align 8
  %call = call ptr @strcpy(ptr %buf, ptr %0)
  %call2 = call i32 (ptr, ...) @printf(ptr @.str, ptr %buf)
  ret void
}

; test3d:  array of [4 x i8]
;          sspreq attribute
; Requires protector.
; Function Attrs: sspreq 
define void @test3d(ptr %a) #2 {
entry:
; LINUX-I386-LABEL: test3d:
; LINUX-I386: mov{{l|q}} %gs:
; LINUX-I386: calll __stack_chk_fail

; LINUX-X64-LABEL: test3d:
; LINUX-X64: mov{{l|q}} %fs:
; LINUX-X64: callq __stack_chk_fail

; LINUX-KERNEL-X64-LABEL: test3d:
; LINUX-KERNEL-X64: mov{{l|q}} %gs:
; LINUX-KERNEL-X64: callq __stack_chk_fail

; DARWIN-X64-LABEL: test3d:
; DARWIN-X64: mov{{l|q}} ___stack_chk_guard
; DARWIN-X64: callq ___stack_chk_fail

; MSVC-I386-LABEL: test3d:
; MSVC-I386: movl ___security_cookie,
; MSVC-I386: calll @__security_check_cookie@4

; MINGW-X64-LABEL: test3d:
; MINGW-X64: mov{{l|q}} .refptr.__stack_chk_guard
; MINGW-X64: callq __stack_chk_fail

  %a.addr = alloca ptr, align 8
  %buf = alloca [4 x i8], align 1
  store ptr %a, ptr %a.addr, align 8
  %0 = load ptr, ptr %a.addr, align 8
  %call = call ptr @strcpy(ptr %buf, ptr %0)
  %call2 = call i32 (ptr, ...) @printf(ptr @.str, ptr %buf)
  ret void
}

; test4a:  struct { [4 x i8] }
;          no ssp attribute
; Requires no protector.
define void @test4a(ptr %a) {
entry:
; LINUX-I386-LABEL: test4a:
; LINUX-I386-NOT: calll __stack_chk_fail
; LINUX-I386: .cfi_endproc

; LINUX-X64-LABEL: test4a:
; LINUX-X64-NOT: callq __stack_chk_fail
; LINUX-X64: .cfi_endproc

; LINUX-KERNEL-X64-LABEL: test4a:
; LINUX-KERNEL-X64-NOT: callq __stack_chk_fail
; LINUX-KERNEL-X64: .cfi_endproc

; DARWIN-X64-LABEL: test4a:
; DARWIN-X64-NOT: callq ___stack_chk_fail
; DARWIN-X64: .cfi_endproc

; MSVC-I386-LABEL: test4a:
; MSVC-I386-NOT: calll @__security_check_cookie@4
; MSVC-I386: retl

; MINGW-X64-LABEL: test4a:
; MINGW-X64-NOT: callq __stack_chk_fail
; MINGW-X64: .seh_endproc

  %a.addr = alloca ptr, align 8
  %b = alloca %struct.foo.0, align 1
  store ptr %a, ptr %a.addr, align 8
  %0 = load ptr, ptr %a.addr, align 8
  %call = call ptr @strcpy(ptr %b, ptr %0)
  %call3 = call i32 (ptr, ...) @printf(ptr @.str, ptr %b)
  ret void
}

; test4b:  struct { [4 x i8] }
;          ssp attribute
; Requires no protector.
; Function Attrs: ssp
define void @test4b(ptr %a) #0 {
entry:
; LINUX-I386-LABEL: test4b:
; LINUX-I386-NOT: calll __stack_chk_fail
; LINUX-I386: .cfi_endproc

; LINUX-X64-LABEL: test4b:
; LINUX-X64-NOT: callq __stack_chk_fail
; LINUX-X64: .cfi_endproc

; LINUX-KERNEL-X64-LABEL: test4b:
; LINUX-KERNEL-X64-NOT: callq __stack_chk_fail
; LINUX-KERNEL-X64: .cfi_endproc

; DARWIN-X64-LABEL: test4b:
; DARWIN-X64-NOT: callq ___stack_chk_fail
; DARWIN-X64: .cfi_endproc

; MSVC-I386-LABEL: test4b:
; MSVC-I386-NOT: calll @__security_check_cookie@4
; MSVC-I386: retl

; MINGW-X64-LABEL: test4b:
; MINGW-X64-NOT: callq __stack_chk_fail
; MINGW-X64: .seh_endproc

  %a.addr = alloca ptr, align 8
  %b = alloca %struct.foo.0, align 1
  store ptr %a, ptr %a.addr, align 8
  %0 = load ptr, ptr %a.addr, align 8
  %call = call ptr @strcpy(ptr %b, ptr %0)
  %call3 = call i32 (ptr, ...) @printf(ptr @.str, ptr %b)
  ret void
}

; test4c:  struct { [4 x i8] }
;          sspstrong attribute
; Requires protector.
; Function Attrs: sspstrong 
define void @test4c(ptr %a) #1 {
entry:
; LINUX-I386-LABEL: test4c:
; LINUX-I386: mov{{l|q}} %gs:
; LINUX-I386: calll __stack_chk_fail

; LINUX-X64-LABEL: test4c:
; LINUX-X64: mov{{l|q}} %fs:
; LINUX-X64: callq __stack_chk_fail

; LINUX-KERNEL-X64-LABEL: test4c:
; LINUX-KERNEL-X64: mov{{l|q}} %gs:
; LINUX-KERNEL-X64: callq __stack_chk_fail

; DARWIN-X64-LABEL: test4c:
; DARWIN-X64: mov{{l|q}} ___stack_chk_guard
; DARWIN-X64: callq ___stack_chk_fail

; MSVC-I386-LABEL: test4c:
; MSVC-I386: movl ___security_cookie,
; MSVC-I386: calll @__security_check_cookie@4

; MINGW-X64-LABEL: test4c:
; MINGW-X64: mov{{l|q}} .refptr.__stack_chk_guard
; MINGW-X64: callq __stack_chk_fail

  %a.addr = alloca ptr, align 8
  %b = alloca %struct.foo.0, align 1
  store ptr %a, ptr %a.addr, align 8
  %0 = load ptr, ptr %a.addr, align 8
  %call = call ptr @strcpy(ptr %b, ptr %0)
  %call3 = call i32 (ptr, ...) @printf(ptr @.str, ptr %b)
  ret void
}

; test4d:  struct { [4 x i8] }
;          sspreq attribute
; Requires protector.
; Function Attrs: sspreq 
define void @test4d(ptr %a) #2 {
entry:
; LINUX-I386-LABEL: test4d:
; LINUX-I386: mov{{l|q}} %gs:
; LINUX-I386: calll __stack_chk_fail

; LINUX-X64-LABEL: test4d:
; LINUX-X64: mov{{l|q}} %fs:
; LINUX-X64: callq __stack_chk_fail

; LINUX-KERNEL-X64-LABEL: test4d:
; LINUX-KERNEL-X64: mov{{l|q}} %gs:
; LINUX-KERNEL-X64: callq __stack_chk_fail

; DARWIN-X64-LABEL: test4d:
; DARWIN-X64: mov{{l|q}} ___stack_chk_guard
; DARWIN-X64: callq ___stack_chk_fail

; MSVC-I386-LABEL: test4d:
; MSVC-I386: movl ___security_cookie,
; MSVC-I386: calll @__security_check_cookie@4

; MINGW-X64-LABEL: test4d:
; MINGW-X64: mov{{l|q}} .refptr.__stack_chk_guard
; MINGW-X64: callq __stack_chk_fail

  %a.addr = alloca ptr, align 8
  %b = alloca %struct.foo.0, align 1
  store ptr %a, ptr %a.addr, align 8
  %0 = load ptr, ptr %a.addr, align 8
  %call = call ptr @strcpy(ptr %b, ptr %0)
  %call3 = call i32 (ptr, ...) @printf(ptr @.str, ptr %b)
  ret void
}

; test5a:  no arrays / no nested arrays
;          no ssp attribute
; Requires no protector.
define void @test5a(ptr %a) {
entry:
; LINUX-I386-LABEL: test5a:
; LINUX-I386-NOT: calll __stack_chk_fail
; LINUX-I386: .cfi_endproc

; LINUX-X64-LABEL: test5a:
; LINUX-X64-NOT: callq __stack_chk_fail
; LINUX-X64: .cfi_endproc

; LINUX-KERNEL-X64-LABEL: test5a:
; LINUX-KERNEL-X64-NOT: callq __stack_chk_fail
; LINUX-KERNEL-X64: .cfi_endproc

; DARWIN-X64-LABEL: test5a:
; DARWIN-X64-NOT: callq ___stack_chk_fail
; DARWIN-X64: .cfi_endproc

; MSVC-I386-LABEL: test5a:
; MSVC-I386-NOT: calll @__security_check_cookie@4
; MSVC-I386: retl

; MINGW-X64-LABEL: test5a:
; MINGW-X64-NOT: callq __stack_chk_fail
; MINGW-X64: .seh_endproc

  %a.addr = alloca ptr, align 8
  store ptr %a, ptr %a.addr, align 8
  %0 = load ptr, ptr %a.addr, align 8
  %call = call i32 (ptr, ...) @printf(ptr @.str, ptr %0)
  ret void
}

; test5b:  no arrays / no nested arrays
;          ssp attribute
; Requires no protector.
; Function Attrs: ssp
define void @test5b(ptr %a) #0 {
entry:
; LINUX-I386-LABEL: test5b:
; LINUX-I386-NOT: calll __stack_chk_fail
; LINUX-I386: .cfi_endproc

; LINUX-X64-LABEL: test5b:
; LINUX-X64-NOT: callq __stack_chk_fail
; LINUX-X64: .cfi_endproc

; LINUX-KERNEL-X64-LABEL: test5b:
; LINUX-KERNEL-X64-NOT: callq __stack_chk_fail
; LINUX-KERNEL-X64: .cfi_endproc

; DARWIN-X64-LABEL: test5b:
; DARWIN-X64-NOT: callq ___stack_chk_fail
; DARWIN-X64: .cfi_endproc

; MSVC-I386-LABEL: test5b:
; MSVC-I386-NOT: calll @__security_check_cookie@4
; MSVC-I386: retl

; MINGW-X64-LABEL: test5b:
; MINGW-X64-NOT: callq __stack_chk_fail
; MINGW-X64: .seh_endproc

  %a.addr = alloca ptr, align 8
  store ptr %a, ptr %a.addr, align 8
  %0 = load ptr, ptr %a.addr, align 8
  %call = call i32 (ptr, ...) @printf(ptr @.str, ptr %0)
  ret void
}

; test5c:  no arrays / no nested arrays
;          sspstrong attribute
; Requires no protector.
; Function Attrs: sspstrong 
define void @test5c(ptr %a) #1 {
entry:
; LINUX-I386-LABEL: test5c:
; LINUX-I386-NOT: calll __stack_chk_fail
; LINUX-I386: .cfi_endproc

; LINUX-X64-LABEL: test5c:
; LINUX-X64-NOT: callq __stack_chk_fail
; LINUX-X64: .cfi_endproc

; LINUX-KERNEL-X64-LABEL: test5c:
; LINUX-KERNEL-X64-NOT: callq __stack_chk_fail
; LINUX-KERNEL-X64: .cfi_endproc

; DARWIN-X64-LABEL: test5c:
; DARWIN-X64-NOT: callq ___stack_chk_fail
; DARWIN-X64: .cfi_endproc

; MSVC-I386-LABEL: test5c:
; MSVC-I386-NOT: calll @__security_check_cookie@4
; MSVC-I386: retl

; MINGW-X64-LABEL: test5c:
; MINGW-X64-NOT: callq __stack_chk_fail
; MINGW-X64: .seh_endproc

  %a.addr = alloca ptr, align 8
  store ptr %a, ptr %a.addr, align 8
  %0 = load ptr, ptr %a.addr, align 8
  %call = call i32 (ptr, ...) @printf(ptr @.str, ptr %0)
  ret void
}

; test5d:  no arrays / no nested arrays
;          sspreq attribute
; Requires protector.
; Function Attrs: sspreq 
define void @test5d(ptr %a) #2 {
entry:
; LINUX-I386-LABEL: test5d:
; LINUX-I386: mov{{l|q}} %gs:
; LINUX-I386: calll __stack_chk_fail

; LINUX-X64-LABEL: test5d:
; LINUX-X64: mov{{l|q}} %fs:
; LINUX-X64: callq __stack_chk_fail

; LINUX-KERNEL-X64-LABEL: test5d:
; LINUX-KERNEL-X64: mov{{l|q}} %gs:
; LINUX-KERNEL-X64: callq __stack_chk_fail

; DARWIN-X64-LABEL: test5d:
; DARWIN-X64: mov{{l|q}} ___stack_chk_guard
; DARWIN-X64: callq ___stack_chk_fail

; MSVC-I386-LABEL: test5d:
; MSVC-I386: movl ___security_cookie,
; MSVC-I386: calll @__security_check_cookie@4

; MINGW-X64-LABEL: test5d:
; MINGW-X64: mov{{l|q}} .refptr.__stack_chk_guard
; MINGW-X64: callq __stack_chk_fail

  %a.addr = alloca ptr, align 8
  store ptr %a, ptr %a.addr, align 8
  %0 = load ptr, ptr %a.addr, align 8
  %call = call i32 (ptr, ...) @printf(ptr @.str, ptr %0)
  ret void
}

; test6a:  Address-of local taken (j = &a)
;          no ssp attribute
; Requires no protector.
define void @test6a() {
entry:
; LINUX-I386-LABEL: test6a:
; LINUX-I386-NOT: calll __stack_chk_fail
; LINUX-I386: .cfi_endproc

; LINUX-X64-LABEL: test6a:
; LINUX-X64-NOT: callq __stack_chk_fail
; LINUX-X64: .cfi_endproc

; LINUX-KERNEL-X64-LABEL: test6a:
; LINUX-KERNEL-X64-NOT: callq __stack_chk_fail
; LINUX-KERNEL-X64: .cfi_endproc

; DARWIN-X64-LABEL: test6a:
; DARWIN-X64-NOT: callq ___stack_chk_fail
; DARWIN-X64: .cfi_endproc

; MSVC-I386-LABEL: test6a:
; MSVC-I386-NOT: calll @__security_check_cookie@4
; MSVC-I386: retl

; MINGW-X64-LABEL: test6a:
; MINGW-X64-NOT: callq __stack_chk_fail
; MINGW-X64: .seh_endproc

  %retval = alloca i32, align 4
  %a = alloca i32, align 4
  %j = alloca ptr, align 8
  store i32 0, ptr %retval
  %0 = load i32, ptr %a, align 4
  %add = add nsw i32 %0, 1
  store i32 %add, ptr %a, align 4
  store ptr %a, ptr %j, align 8
  ret void
}

; test6b:  Address-of local taken (j = &a)
;          ssp attribute
; Requires no protector.
; Function Attrs: ssp
define void @test6b() #0 {
entry:
; LINUX-I386-LABEL: test6b:
; LINUX-I386-NOT: calll __stack_chk_fail
; LINUX-I386: .cfi_endproc

; LINUX-X64-LABEL: test6b:
; LINUX-X64-NOT: callq __stack_chk_fail
; LINUX-X64: .cfi_endproc

; LINUX-KERNEL-X64-LABEL: test6b:
; LINUX-KERNEL-X64-NOT: callq __stack_chk_fail
; LINUX-KERNEL-X64: .cfi_endproc

; DARWIN-X64-LABEL: test6b:
; DARWIN-X64-NOT: callq ___stack_chk_fail
; DARWIN-X64: .cfi_endproc

; MSVC-I386-LABEL: test6b:
; MSVC-I386-NOT: calll @__security_check_cookie@4
; MSVC-I386: retl

; MINGW-X64-LABEL: test6b:
; MINGW-X64-NOT: callq __stack_chk_fail
; MINGW-X64: .seh_endproc

  %retval = alloca i32, align 4
  %a = alloca i32, align 4
  %j = alloca ptr, align 8
  store i32 0, ptr %retval
  %0 = load i32, ptr %a, align 4
  %add = add nsw i32 %0, 1
  store i32 %add, ptr %a, align 4
  store ptr %a, ptr %j, align 8
  ret void
}

; test6c:  Address-of local taken (j = &a)
;          sspstrong attribute
; Requires protector.
; Function Attrs: sspstrong 
define void @test6c() #1 {
entry:
; LINUX-I386-LABEL: test6c:
; LINUX-I386: mov{{l|q}} %gs:
; LINUX-I386: calll __stack_chk_fail

; LINUX-X64-LABEL: test6c:
; LINUX-X64: mov{{l|q}} %fs:
; LINUX-X64: callq __stack_chk_fail

; LINUX-KERNEL-X64-LABEL: test6c:
; LINUX-KERNEL-X64: mov{{l|q}} %gs:
; LINUX-KERNEL-X64: callq __stack_chk_fail

; DARWIN-X64-LABEL: test6c:
; DARWIN-X64: mov{{l|q}} ___stack_chk_guard
; DARWIN-X64: callq ___stack_chk_fail

; MSVC-I386-LABEL: test6c:
; MSVC-I386: movl ___security_cookie,
; MSVC-I386: calll @__security_check_cookie@4

; MINGW-X64-LABEL: test6c:
; MINGW-X64: mov{{l|q}} .refptr.__stack_chk_guard
; MINGW-X64: callq __stack_chk_fail

  %retval = alloca i32, align 4
  %a = alloca i32, align 4
  %j = alloca ptr, align 8
  store i32 0, ptr %retval
  %0 = load i32, ptr %a, align 4
  %add = add nsw i32 %0, 1
  store i32 %add, ptr %a, align 4
  store ptr %a, ptr %j, align 8
  ret void
}

; test6d:  Address-of local taken (j = &a)
;          sspreq attribute
; Requires protector.
; Function Attrs: sspreq 
define void @test6d() #2 {
entry:
; LINUX-I386-LABEL: test6d:
; LINUX-I386: mov{{l|q}} %gs:
; LINUX-I386: calll __stack_chk_fail

; LINUX-X64-LABEL: test6d:
; LINUX-X64: mov{{l|q}} %fs:
; LINUX-X64: callq __stack_chk_fail

; LINUX-KERNEL-X64-LABEL: test6d:
; LINUX-KERNEL-X64: mov{{l|q}} %gs:
; LINUX-KERNEL-X64: callq __stack_chk_fail

; DARWIN-X64-LABEL: test6d:
; DARWIN-X64: mov{{l|q}} ___stack_chk_guard
; DARWIN-X64: callq ___stack_chk_fail

; MSVC-I386-LABEL: test6d:
; MSVC-I386: movl ___security_cookie,
; MSVC-I386: calll @__security_check_cookie@4

; MINGW-X64-LABEL: test6d:
; MINGW-X64: mov{{l|q}} .refptr.__stack_chk_guard
; MINGW-X64: callq __stack_chk_fail

  %retval = alloca i32, align 4
  %a = alloca i32, align 4
  %j = alloca ptr, align 8
  store i32 0, ptr %retval
  %0 = load i32, ptr %a, align 4
  %add = add nsw i32 %0, 1
  store i32 %add, ptr %a, align 4
  store ptr %a, ptr %j, align 8
  ret void
}

; test7a:  PtrToInt Cast
;          no ssp attribute
; Requires no protector.
define void @test7a()  {
entry:
; LINUX-I386-LABEL: test7a:
; LINUX-I386-NOT: calll __stack_chk_fail
; LINUX-I386: .cfi_endproc

; LINUX-X64-LABEL: test7a:
; LINUX-X64-NOT: callq __stack_chk_fail
; LINUX-X64: .cfi_endproc

; LINUX-KERNEL-X64-LABEL: test7a:
; LINUX-KERNEL-X64-NOT: callq __stack_chk_fail
; LINUX-KERNEL-X64: .cfi_endproc

; DARWIN-X64-LABEL: test7a:
; DARWIN-X64-NOT: callq ___stack_chk_fail
; DARWIN-X64: .cfi_endproc

; MSVC-I386-LABEL: test7a:
; MSVC-I386-NOT: calll @__security_check_cookie@4
; MSVC-I386: retl

; MINGW-X64-LABEL: test7a:
; MINGW-X64-NOT: callq __stack_chk_fail
; MINGW-X64: .seh_endproc

  %a = alloca i32, align 4
  %0 = ptrtoint ptr %a to i64
  %call = call i32 (ptr, ...) @printf(ptr @.str, i64 %0)
  ret void
}

; test7b:  PtrToInt Cast
;          ssp attribute
; Requires no protector.
; Function Attrs: ssp 
define void @test7b() #0 {
entry:
; LINUX-I386-LABEL: test7b:
; LINUX-I386-NOT: calll __stack_chk_fail
; LINUX-I386: .cfi_endproc

; LINUX-X64-LABEL: test7b:
; LINUX-X64-NOT: callq __stack_chk_fail
; LINUX-X64: .cfi_endproc

; LINUX-KERNEL-X64-LABEL: test7b:
; LINUX-KERNEL-X64-NOT: callq __stack_chk_fail
; LINUX-KERNEL-X64: .cfi_endproc

; DARWIN-X64-LABEL: test7b:
; DARWIN-X64-NOT: callq ___stack_chk_fail
; DARWIN-X64: .cfi_endproc

; MSVC-I386-LABEL: test7b:
; MSVC-I386-NOT: calll @__security_check_cookie@4
; MSVC-I386: retl

; MINGW-X64-LABEL: test7b:
; MINGW-X64-NOT: callq __stack_chk_fail
; MINGW-X64: .seh_endproc

  %a = alloca i32, align 4
  %0 = ptrtoint ptr %a to i64
  %call = call i32 (ptr, ...) @printf(ptr @.str, i64 %0)
  ret void
}

; test7c:  PtrToInt Cast
;          sspstrong attribute
; Requires protector.
; Function Attrs: sspstrong 
define void @test7c() #1 {
entry:
; LINUX-I386-LABEL: test7c:
; LINUX-I386: mov{{l|q}} %gs:
; LINUX-I386: calll __stack_chk_fail

; LINUX-X64-LABEL: test7c:
; LINUX-X64: mov{{l|q}} %fs:
; LINUX-X64: callq __stack_chk_fail

; LINUX-KERNEL-X64-LABEL: test7c:
; LINUX-KERNEL-X64: mov{{l|q}} %gs:
; LINUX-KERNEL-X64: callq __stack_chk_fail

; DARWIN-X64-LABEL: test7c:
; DARWIN-X64: mov{{l|q}} ___stack_chk_guard
; DARWIN-X64: callq ___stack_chk_fail

; MSVC-I386-LABEL: test7c:
; MSVC-I386: movl ___security_cookie,
; MSVC-I386: calll @__security_check_cookie@4

; MINGW-X64-LABEL: test7c:
; MINGW-X64: mov{{l|q}} .refptr.__stack_chk_guard
; MINGW-X64: .seh_endproc

  %a = alloca i32, align 4
  %0 = ptrtoint ptr %a to i64
  %call = call i32 (ptr, ...) @printf(ptr @.str, i64 %0)
  ret void
}

; test7d:  PtrToInt Cast
;          sspreq attribute
; Requires protector.
; Function Attrs: sspreq 
define void @test7d() #2 {
entry:
; LINUX-I386-LABEL: test7d:
; LINUX-I386: mov{{l|q}} %gs:
; LINUX-I386: calll __stack_chk_fail

; LINUX-X64-LABEL: test7d:
; LINUX-X64: mov{{l|q}} %fs:
; LINUX-X64: callq __stack_chk_fail

; LINUX-KERNEL-X64-LABEL: test7d:
; LINUX-KERNEL-X64: mov{{l|q}} %gs:
; LINUX-KERNEL-X64: callq __stack_chk_fail

; DARWIN-X64-LABEL: test7d:
; DARWIN-X64: mov{{l|q}} ___stack_chk_guard
; DARWIN-X64: callq ___stack_chk_fail

; MSVC-I386-LABEL: test7d:
; MSVC-I386: movl ___security_cookie,
; MSVC-I386: calll @__security_check_cookie@4

; MINGW-X64-LABEL: test7d:
; MINGW-X64: mov{{l|q}} .refptr.__stack_chk_guard
; MINGW-X64: callq __stack_chk_fail

  %a = alloca i32, align 4
  %0 = ptrtoint ptr %a to i64
  %call = call i32 (ptr, ...) @printf(ptr @.str, i64 %0)
  ret void
}

; test8a:  Passing addr-of to function call
;          no ssp attribute
; Requires no protector.
define void @test8a() {
entry:
; LINUX-I386-LABEL: test8a:
; LINUX-I386-NOT: calll __stack_chk_fail
; LINUX-I386: .cfi_endproc

; LINUX-X64-LABEL: test8a:
; LINUX-X64-NOT: callq __stack_chk_fail
; LINUX-X64: .cfi_endproc

; LINUX-KERNEL-X64-LABEL: test8a:
; LINUX-KERNEL-X64-NOT: callq __stack_chk_fail
; LINUX-KERNEL-X64: .cfi_endproc

; DARWIN-X64-LABEL: test8a:
; DARWIN-X64-NOT: callq ___stack_chk_fail
; DARWIN-X64: .cfi_endproc

; MSVC-I386-LABEL: test8a:
; MSVC-I386-NOT: calll @__security_check_cookie@4
; MSVC-I386: retl

; MINGW-X64-LABEL: test8a:
; MINGW-X64-NOT: callq __stack_chk_fail
; MINGW-X64: .seh_endproc

  %b = alloca i32, align 4
  call void @funcall(ptr %b)
  ret void
}

; test8b:  Passing addr-of to function call
;          ssp attribute
; Requires no protector.
; Function Attrs: ssp
define void @test8b() #0 {
entry:
; LINUX-I386-LABEL: test8b:
; LINUX-I386-NOT: calll __stack_chk_fail
; LINUX-I386: .cfi_endproc

; LINUX-X64-LABEL: test8b:
; LINUX-X64-NOT: callq __stack_chk_fail
; LINUX-X64: .cfi_endproc

; LINUX-KERNEL-X64-LABEL: test8b:
; LINUX-KERNEL-X64-NOT: callq __stack_chk_fail
; LINUX-KERNEL-X64: .cfi_endproc

; DARWIN-X64-LABEL: test8b:
; DARWIN-X64-NOT: callq ___stack_chk_fail
; DARWIN-X64: .cfi_endproc

; MSVC-I386-LABEL: test8b:
; MSVC-I386-NOT: calll @__security_check_cookie@4
; MSVC-I386: retl

; MINGW-X64-LABEL: test8b:
; MINGW-X64-NOT: callq __stack_chk_fail
; MINGW-X64: .seh_endproc

  %b = alloca i32, align 4
  call void @funcall(ptr %b)
  ret void
}

; test8c:  Passing addr-of to function call
;          sspstrong attribute
; Requires protector.
; Function Attrs: sspstrong 
define void @test8c() #1 {
entry:
; LINUX-I386-LABEL: test8c:
; LINUX-I386: mov{{l|q}} %gs:
; LINUX-I386: calll __stack_chk_fail

; LINUX-X64-LABEL: test8c:
; LINUX-X64: mov{{l|q}} %fs:
; LINUX-X64: callq __stack_chk_fail

; LINUX-KERNEL-X64-LABEL: test8c:
; LINUX-KERNEL-X64: mov{{l|q}} %gs:
; LINUX-KERNEL-X64: callq __stack_chk_fail

; DARWIN-X64-LABEL: test8c:
; DARWIN-X64: mov{{l|q}} ___stack_chk_guard
; DARWIN-X64: callq ___stack_chk_fail

; MSVC-I386-LABEL: test8c:
; MSVC-I386: movl ___security_cookie,
; MSVC-I386: calll @__security_check_cookie@4

; MINGW-X64-LABEL: test8c:
; MINGW-X64: mov{{l|q}} .refptr.__stack_chk_guard
; MINGW-X64: callq __stack_chk_fail

  %b = alloca i32, align 4
  call void @funcall(ptr %b)
  ret void
}

; test8d:  Passing addr-of to function call
;          sspreq attribute
; Requires protector.
; Function Attrs: sspreq 
define void @test8d() #2 {
entry:
; LINUX-I386-LABEL: test8d:
; LINUX-I386: mov{{l|q}} %gs:
; LINUX-I386: calll __stack_chk_fail

; LINUX-X64-LABEL: test8d:
; LINUX-X64: mov{{l|q}} %fs:
; LINUX-X64: callq __stack_chk_fail

; LINUX-KERNEL-X64-LABEL: test8d:
; LINUX-KERNEL-X64: mov{{l|q}} %gs:
; LINUX-KERNEL-X64: callq __stack_chk_fail

; DARWIN-X64-LABEL: test8d:
; DARWIN-X64: mov{{l|q}} ___stack_chk_guard
; DARWIN-X64: callq ___stack_chk_fail

; MSVC-I386-LABEL: test8d:
; MSVC-I386: movl ___security_cookie,
; MSVC-I386: calll @__security_check_cookie@4

; MINGW-X64-LABEL: test8d:
; MINGW-X64: mov{{l|q}} .refptr.__stack_chk_guard
; MINGW-X64: callq __stack_chk_fail

  %b = alloca i32, align 4
  call void @funcall(ptr %b)
  ret void
}

; test9a:  Addr-of in select instruction
;          no ssp attribute
; Requires no protector.
define void @test9a() {
entry:
; LINUX-I386-LABEL: test9a:
; LINUX-I386-NOT: calll __stack_chk_fail
; LINUX-I386: .cfi_endproc

; LINUX-X64-LABEL: test9a:
; LINUX-X64-NOT: callq __stack_chk_fail
; LINUX-X64: .cfi_endproc

; LINUX-KERNEL-X64-LABEL: test9a:
; LINUX-KERNEL-X64-NOT: callq __stack_chk_fail
; LINUX-KERNEL-X64: .cfi_endproc

; DARWIN-X64-LABEL: test9a:
; DARWIN-X64-NOT: callq ___stack_chk_fail
; DARWIN-X64: .cfi_endproc

; MSVC-I386-LABEL: test9a:
; MSVC-I386-NOT: calll @__security_check_cookie@4
; MSVC-I386: retl
  %x = alloca double, align 8
  %call = call double @testi_aux()
  store double %call, ptr %x, align 8
  %cmp2 = fcmp ogt double %call, 0.000000e+00
  %y.1 = select i1 %cmp2, ptr %x, ptr null
  %call2 = call i32 (ptr, ...) @printf(ptr @.str, ptr %y.1)
  ret void
}

; test9b:  Addr-of in select instruction
;          ssp attribute
; Requires no protector.
; Function Attrs: ssp
define void @test9b() #0 {
entry:
; LINUX-I386-LABEL: test9b:
; LINUX-I386-NOT: calll __stack_chk_fail
; LINUX-I386: .cfi_endproc

; LINUX-X64-LABEL: test9b:
; LINUX-X64-NOT: callq __stack_chk_fail
; LINUX-X64: .cfi_endproc

; LINUX-KERNEL-X64-LABEL: test9b:
; LINUX-KERNEL-X64-NOT: callq __stack_chk_fail
; LINUX-KERNEL-X64: .cfi_endproc

; DARWIN-X64-LABEL: test9b:
; DARWIN-X64-NOT: callq ___stack_chk_fail
; DARWIN-X64: .cfi_endproc

; MSVC-I386-LABEL: test9b:
; MSVC-I386-NOT: calll @__security_check_cookie@4
; MSVC-I386: retl
  %x = alloca double, align 8
  %call = call double @testi_aux()
  store double %call, ptr %x, align 8
  %cmp2 = fcmp ogt double %call, 0.000000e+00
  %y.1 = select i1 %cmp2, ptr %x, ptr null
  %call2 = call i32 (ptr, ...) @printf(ptr @.str, ptr %y.1)
  ret void
}

; test9c:  Addr-of in select instruction
;          sspstrong attribute
; Requires protector.
; Function Attrs: sspstrong 
define void @test9c() #1 {
entry:
; LINUX-I386-LABEL: test9c:
; LINUX-I386: mov{{l|q}} %gs:
; LINUX-I386: calll __stack_chk_fail

; LINUX-X64-LABEL: test9c:
; LINUX-X64: mov{{l|q}} %fs:
; LINUX-X64: callq __stack_chk_fail

; LINUX-KERNEL-X64-LABEL: test9c:
; LINUX-KERNEL-X64: mov{{l|q}} %gs:
; LINUX-KERNEL-X64: callq __stack_chk_fail

; DARWIN-X64-LABEL: test9c:
; DARWIN-X64: mov{{l|q}} ___stack_chk_guard
; DARWIN-X64: callq ___stack_chk_fail

; MSVC-I386-LABEL: test9c:
; MSVC-I386: movl ___security_cookie,
; MSVC-I386: calll @__security_check_cookie@4
  %x = alloca double, align 8
  %call = call double @testi_aux()
  store double %call, ptr %x, align 8
  %cmp2 = fcmp ogt double %call, 0.000000e+00
  %y.1 = select i1 %cmp2, ptr %x, ptr null
  %call2 = call i32 (ptr, ...) @printf(ptr @.str, ptr %y.1)
  ret void
}

; test9d:  Addr-of in select instruction
;          sspreq attribute
; Requires protector.
; Function Attrs: sspreq 
define void @test9d() #2 {
entry:
; LINUX-I386-LABEL: test9d:
; LINUX-I386: mov{{l|q}} %gs:
; LINUX-I386: calll __stack_chk_fail

; LINUX-X64-LABEL: test9d:
; LINUX-X64: mov{{l|q}} %fs:
; LINUX-X64: callq __stack_chk_fail

; LINUX-KERNEL-X64-LABEL: test9d:
; LINUX-KERNEL-X64: mov{{l|q}} %gs:
; LINUX-KERNEL-X64: callq __stack_chk_fail

; DARWIN-X64-LABEL: test9d:
; DARWIN-X64: mov{{l|q}} ___stack_chk_guard
; DARWIN-X64: callq ___stack_chk_fail

; MSVC-I386-LABEL: test9d:
; MSVC-I386: movl ___security_cookie,
; MSVC-I386: calll @__security_check_cookie@4
  %x = alloca double, align 8
  %call = call double @testi_aux()
  store double %call, ptr %x, align 8
  %cmp2 = fcmp ogt double %call, 0.000000e+00
  %y.1 = select i1 %cmp2, ptr %x, ptr null
  %call2 = call i32 (ptr, ...) @printf(ptr @.str, ptr %y.1)
  ret void
}

; test10a: Addr-of in phi instruction
;          no ssp attribute
; Requires no protector.
define void @test10a() {
entry:
; LINUX-I386-LABEL: test10a:
; LINUX-I386-NOT: calll __stack_chk_fail
; LINUX-I386: .cfi_endproc

; LINUX-X64-LABEL: test10a:
; LINUX-X64-NOT: callq __stack_chk_fail
; LINUX-X64: .cfi_endproc

; LINUX-KERNEL-X64-LABEL: test10a:
; LINUX-KERNEL-X64-NOT: callq __stack_chk_fail
; LINUX-KERNEL-X64: .cfi_endproc

; DARWIN-X64-LABEL: test10a:
; DARWIN-X64-NOT: callq ___stack_chk_fail
; DARWIN-X64: .cfi_endproc

; MSVC-I386-LABEL: test10a:
; MSVC-I386-NOT: calll @__security_check_cookie@4
; MSVC-I386: retl
  %x = alloca double, align 8
  %call = call double @testi_aux()
  store double %call, ptr %x, align 8
  %cmp = fcmp ogt double %call, 3.140000e+00
  br i1 %cmp, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  %call1 = call double @testi_aux()
  store double %call1, ptr %x, align 8
  br label %if.end4

if.else:                                          ; preds = %entry
  %cmp2 = fcmp ogt double %call, 1.000000e+00
  br i1 %cmp2, label %if.then3, label %if.end4

if.then3:                                         ; preds = %if.else
  br label %if.end4

if.end4:                                          ; preds = %if.else, %if.then3, %if.then
  %y.0 = phi ptr [ null, %if.then ], [ %x, %if.then3 ], [ null, %if.else ]
  %call5 = call i32 (ptr, ...) @printf(ptr @.str, ptr %y.0)
  ret void
}

; test10b: Addr-of in phi instruction
;          ssp attribute
; Requires no protector.
; Function Attrs: ssp
define void @test10b() #0 {
entry:
; LINUX-I386-LABEL: test10b:
; LINUX-I386-NOT: calll __stack_chk_fail
; LINUX-I386: .cfi_endproc

; LINUX-X64-LABEL: test10b:
; LINUX-X64-NOT: callq __stack_chk_fail
; LINUX-X64: .cfi_endproc

; LINUX-KERNEL-X64-LABEL: test10b:
; LINUX-KERNEL-X64-NOT: callq __stack_chk_fail
; LINUX-KERNEL-X64: .cfi_endproc

; DARWIN-X64-LABEL: test10b:
; DARWIN-X64-NOT: callq ___stack_chk_fail
; DARWIN-X64: .cfi_endproc

; MSVC-I386-LABEL: test10b:
; MSVC-I386-NOT: calll @__security_check_cookie@4
; MSVC-I386: retl
  %x = alloca double, align 8
  %call = call double @testi_aux()
  store double %call, ptr %x, align 8
  %cmp = fcmp ogt double %call, 3.140000e+00
  br i1 %cmp, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  %call1 = call double @testi_aux()
  store double %call1, ptr %x, align 8
  br label %if.end4

if.else:                                          ; preds = %entry
  %cmp2 = fcmp ogt double %call, 1.000000e+00
  br i1 %cmp2, label %if.then3, label %if.end4

if.then3:                                         ; preds = %if.else
  br label %if.end4

if.end4:                                          ; preds = %if.else, %if.then3, %if.then
  %y.0 = phi ptr [ null, %if.then ], [ %x, %if.then3 ], [ null, %if.else ]
  %call5 = call i32 (ptr, ...) @printf(ptr @.str, ptr %y.0)
  ret void
}

; test10c: Addr-of in phi instruction
;          sspstrong attribute
; Requires protector.
; Function Attrs: sspstrong 
define void @test10c() #1 {
entry:
; LINUX-I386-LABEL: test10c:
; LINUX-I386: mov{{l|q}} %gs:
; LINUX-I386: calll __stack_chk_fail

; LINUX-X64-LABEL: test10c:
; LINUX-X64: mov{{l|q}} %fs:
; LINUX-X64: callq __stack_chk_fail

; LINUX-KERNEL-X64-LABEL: test10c:
; LINUX-KERNEL-X64: mov{{l|q}} %gs:
; LINUX-KERNEL-X64: callq __stack_chk_fail

; DARWIN-X64-LABEL: test10c:
; DARWIN-X64: mov{{l|q}} ___stack_chk_guard
; DARWIN-X64: callq ___stack_chk_fail

; MSVC-I386-LABEL: test10c:
; MSVC-I386: movl ___security_cookie,
; MSVC-I386: calll @__security_check_cookie@4
  %x = alloca double, align 8
  %call = call double @testi_aux()
  store double %call, ptr %x, align 8
  %cmp = fcmp ogt double %call, 3.140000e+00
  br i1 %cmp, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  %call1 = call double @testi_aux()
  store double %call1, ptr %x, align 8
  br label %if.end4

if.else:                                          ; preds = %entry
  %cmp2 = fcmp ogt double %call, 1.000000e+00
  br i1 %cmp2, label %if.then3, label %if.end4

if.then3:                                         ; preds = %if.else
  br label %if.end4

if.end4:                                          ; preds = %if.else, %if.then3, %if.then
  %y.0 = phi ptr [ null, %if.then ], [ %x, %if.then3 ], [ null, %if.else ]
  %call5 = call i32 (ptr, ...) @printf(ptr @.str, ptr %y.0)
  ret void
}

; test10d: Addr-of in phi instruction
;          sspreq attribute
; Requires protector.
; Function Attrs: sspreq 
define void @test10d() #2 {
entry:
; LINUX-I386-LABEL: test10d:
; LINUX-I386: mov{{l|q}} %gs:
; LINUX-I386: calll __stack_chk_fail

; LINUX-X64-LABEL: test10d:
; LINUX-X64: mov{{l|q}} %fs:
; LINUX-X64: callq __stack_chk_fail

; LINUX-KERNEL-X64-LABEL: test10d:
; LINUX-KERNEL-X64: mov{{l|q}} %gs:
; LINUX-KERNEL-X64: callq __stack_chk_fail

; DARWIN-X64-LABEL: test10d:
; DARWIN-X64: mov{{l|q}} ___stack_chk_guard
; DARWIN-X64: callq ___stack_chk_fail

; MSVC-I386-LABEL: test10d:
; MSVC-I386: movl ___security_cookie,
; MSVC-I386: calll @__security_check_cookie@4
  %x = alloca double, align 8
  %call = call double @testi_aux()
  store double %call, ptr %x, align 8
  %cmp = fcmp ogt double %call, 3.140000e+00
  br i1 %cmp, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  %call1 = call double @testi_aux()
  store double %call1, ptr %x, align 8
  br label %if.end4

if.else:                                          ; preds = %entry
  %cmp2 = fcmp ogt double %call, 1.000000e+00
  br i1 %cmp2, label %if.then3, label %if.end4

if.then3:                                         ; preds = %if.else
  br label %if.end4

if.end4:                                          ; preds = %if.else, %if.then3, %if.then
  %y.0 = phi ptr [ null, %if.then ], [ %x, %if.then3 ], [ null, %if.else ]
  %call5 = call i32 (ptr, ...) @printf(ptr @.str, ptr %y.0)
  ret void
}

; test11a: Addr-of struct element. (GEP followed by store).
;          no ssp attribute
; Requires no protector.
define void @test11a() {
entry:
; LINUX-I386-LABEL: test11a:
; LINUX-I386-NOT: calll __stack_chk_fail
; LINUX-I386: .cfi_endproc

; LINUX-X64-LABEL: test11a:
; LINUX-X64-NOT: callq __stack_chk_fail
; LINUX-X64: .cfi_endproc

; LINUX-KERNEL-X64-LABEL: test11a:
; LINUX-KERNEL-X64-NOT: callq __stack_chk_fail
; LINUX-KERNEL-X64: .cfi_endproc

; DARWIN-X64-LABEL: test11a:
; DARWIN-X64-NOT: callq ___stack_chk_fail
; DARWIN-X64: .cfi_endproc

; MSVC-I386-LABEL: test11a:
; MSVC-I386-NOT: calll @__security_check_cookie@4
; MSVC-I386: retl
  %c = alloca %struct.pair, align 4
  %b = alloca ptr, align 8
  %y = getelementptr inbounds %struct.pair, ptr %c, i32 0, i32 1
  store ptr %y, ptr %b, align 8
  %0 = load ptr, ptr %b, align 8
  %call = call i32 (ptr, ...) @printf(ptr @.str, ptr %0)
  ret void
}

; test11b: Addr-of struct element. (GEP followed by store).
;          ssp attribute
; Requires no protector.
; Function Attrs: ssp
define void @test11b() #0 {
entry:
; LINUX-I386-LABEL: test11b:
; LINUX-I386-NOT: calll __stack_chk_fail
; LINUX-I386: .cfi_endproc

; LINUX-X64-LABEL: test11b:
; LINUX-X64-NOT: callq __stack_chk_fail
; LINUX-X64: .cfi_endproc

; LINUX-KERNEL-X64-LABEL: test11b:
; LINUX-KERNEL-X64-NOT: callq __stack_chk_fail
; LINUX-KERNEL-X64: .cfi_endproc

; DARWIN-X64-LABEL: test11b:
; DARWIN-X64-NOT: callq ___stack_chk_fail
; DARWIN-X64: .cfi_endproc

; MSVC-I386-LABEL: test11b:
; MSVC-I386-NOT: calll @__security_check_cookie@4
; MSVC-I386: retl
  %c = alloca %struct.pair, align 4
  %b = alloca ptr, align 8
  %y = getelementptr inbounds %struct.pair, ptr %c, i32 0, i32 1
  store ptr %y, ptr %b, align 8
  %0 = load ptr, ptr %b, align 8
  %call = call i32 (ptr, ...) @printf(ptr @.str, ptr %0)
  ret void
}

; test11c: Addr-of struct element. (GEP followed by store).
;          sspstrong attribute
; Requires protector.
; Function Attrs: sspstrong 
define void @test11c() #1 {
entry:
; LINUX-I386-LABEL: test11c:
; LINUX-I386: mov{{l|q}} %gs:
; LINUX-I386: calll __stack_chk_fail

; LINUX-X64-LABEL: test11c:
; LINUX-X64: mov{{l|q}} %fs:
; LINUX-X64: callq __stack_chk_fail

; LINUX-KERNEL-X64-LABEL: test11c:
; LINUX-KERNEL-X64: mov{{l|q}} %gs:
; LINUX-KERNEL-X64: callq __stack_chk_fail

; DARWIN-X64-LABEL: test11c:
; DARWIN-X64: mov{{l|q}} ___stack_chk_guard
; DARWIN-X64: callq ___stack_chk_fail

; MSVC-I386-LABEL: test11c:
; MSVC-I386: movl ___security_cookie,
; MSVC-I386: calll @__security_check_cookie@4
  %c = alloca %struct.pair, align 4
  %b = alloca ptr, align 8
  %y = getelementptr inbounds %struct.pair, ptr %c, i32 0, i32 1
  store ptr %y, ptr %b, align 8
  %0 = load ptr, ptr %b, align 8
  %call = call i32 (ptr, ...) @printf(ptr @.str, ptr %0)
  ret void
}

; test11d: Addr-of struct element. (GEP followed by store).
;          sspreq attribute
; Requires protector.
; Function Attrs: sspreq 
define void @test11d() #2 {
entry:
; LINUX-I386-LABEL: test11d:
; LINUX-I386: mov{{l|q}} %gs:
; LINUX-I386: calll __stack_chk_fail

; LINUX-X64-LABEL: test11d:
; LINUX-X64: mov{{l|q}} %fs:
; LINUX-X64: callq __stack_chk_fail

; LINUX-KERNEL-X64-LABEL: test11d:
; LINUX-KERNEL-X64: mov{{l|q}} %gs:
; LINUX-KERNEL-X64: callq __stack_chk_fail

; DARWIN-X64-LABEL: test11d:
; DARWIN-X64: mov{{l|q}} ___stack_chk_guard
; DARWIN-X64: callq ___stack_chk_fail

; MSVC-I386-LABEL: test11d:
; MSVC-I386: movl ___security_cookie,
; MSVC-I386: calll @__security_check_cookie@4
  %c = alloca %struct.pair, align 4
  %b = alloca ptr, align 8
  %y = getelementptr inbounds %struct.pair, ptr %c, i32 0, i32 1
  store ptr %y, ptr %b, align 8
  %0 = load ptr, ptr %b, align 8
  %call = call i32 (ptr, ...) @printf(ptr @.str, ptr %0)
  ret void
}

; test12a: Addr-of struct element, GEP followed by ptrtoint.
;          no ssp attribute
; Requires no protector.
define void @test12a() {
entry:
; LINUX-I386-LABEL: test12a:
; LINUX-I386-NOT: calll __stack_chk_fail
; LINUX-I386: .cfi_endproc

; LINUX-X64-LABEL: test12a:
; LINUX-X64-NOT: callq __stack_chk_fail
; LINUX-X64: .cfi_endproc

; LINUX-KERNEL-X64-LABEL: test12a:
; LINUX-KERNEL-X64-NOT: callq __stack_chk_fail
; LINUX-KERNEL-X64: .cfi_endproc

; DARWIN-X64-LABEL: test12a:
; DARWIN-X64-NOT: callq ___stack_chk_fail
; DARWIN-X64: .cfi_endproc

; MSVC-I386-LABEL: test12a:
; MSVC-I386-NOT: calll @__security_check_cookie@4
; MSVC-I386: retl
  %c = alloca %struct.pair, align 4
  %b = alloca ptr, align 8
  %y = getelementptr inbounds %struct.pair, ptr %c, i32 0, i32 1
  %0 = ptrtoint ptr %y to i64
  %call = call i32 (ptr, ...) @printf(ptr @.str, i64 %0)
  ret void
}

; test12b: Addr-of struct element, GEP followed by ptrtoint.
;          ssp attribute
; Requires no protector.
; Function Attrs: ssp
define void @test12b() #0 {
entry:
; LINUX-I386-LABEL: test12b:
; LINUX-I386-NOT: calll __stack_chk_fail
; LINUX-I386: .cfi_endproc

; LINUX-X64-LABEL: test12b:
; LINUX-X64-NOT: callq __stack_chk_fail
; LINUX-X64: .cfi_endproc

; LINUX-KERNEL-X64-LABEL: test12b:
; LINUX-KERNEL-X64-NOT: callq __stack_chk_fail
; LINUX-KERNEL-X64: .cfi_endproc

; DARWIN-X64-LABEL: test12b:
; DARWIN-X64-NOT: callq ___stack_chk_fail
; DARWIN-X64: .cfi_endproc

; MSVC-I386-LABEL: test12b:
; MSVC-I386-NOT: calll @__security_check_cookie@4
; MSVC-I386: retl
  %c = alloca %struct.pair, align 4
  %b = alloca ptr, align 8
  %y = getelementptr inbounds %struct.pair, ptr %c, i32 0, i32 1
  %0 = ptrtoint ptr %y to i64
  %call = call i32 (ptr, ...) @printf(ptr @.str, i64 %0)
  ret void
}

; test12c: Addr-of struct element, GEP followed by ptrtoint.
;          sspstrong attribute
; Function Attrs: sspstrong 
define void @test12c() #1 {
entry:
; LINUX-I386-LABEL: test12c:
; LINUX-I386: mov{{l|q}} %gs:
; LINUX-I386: calll __stack_chk_fail

; LINUX-X64-LABEL: test12c:
; LINUX-X64: mov{{l|q}} %fs:
; LINUX-X64: callq __stack_chk_fail

; LINUX-KERNEL-X64-LABEL: test12c:
; LINUX-KERNEL-X64: mov{{l|q}} %gs:
; LINUX-KERNEL-X64: callq __stack_chk_fail

; DARWIN-X64-LABEL: test12c:
; DARWIN-X64: mov{{l|q}} ___stack_chk_guard
; DARWIN-X64: callq ___stack_chk_fail

; MSVC-I386-LABEL: test12c:
; MSVC-I386: movl ___security_cookie,
; MSVC-I386: calll @__security_check_cookie@4
  %c = alloca %struct.pair, align 4
  %b = alloca ptr, align 8
  %y = getelementptr inbounds %struct.pair, ptr %c, i32 0, i32 1
  %0 = ptrtoint ptr %y to i64
  %call = call i32 (ptr, ...) @printf(ptr @.str, i64 %0)
  ret void
}

; test12d: Addr-of struct element, GEP followed by ptrtoint.
;          sspreq attribute
; Requires protector.
; Function Attrs: sspreq 
define void @test12d() #2 {
entry:
; LINUX-I386-LABEL: test12d:
; LINUX-I386: mov{{l|q}} %gs:
; LINUX-I386: calll __stack_chk_fail

; LINUX-X64-LABEL: test12d:
; LINUX-X64: mov{{l|q}} %fs:
; LINUX-X64: callq __stack_chk_fail

; LINUX-KERNEL-X64-LABEL: test12d:
; LINUX-KERNEL-X64: mov{{l|q}} %gs:
; LINUX-KERNEL-X64: callq __stack_chk_fail

; DARWIN-X64-LABEL: test12d:
; DARWIN-X64: mov{{l|q}} ___stack_chk_guard
; DARWIN-X64: callq ___stack_chk_fail

; MSVC-I386-LABEL: test12d:
; MSVC-I386: movl ___security_cookie,
; MSVC-I386: calll @__security_check_cookie@4
  %c = alloca %struct.pair, align 4
  %b = alloca ptr, align 8
  %y = getelementptr inbounds %struct.pair, ptr %c, i32 0, i32 1
  %0 = ptrtoint ptr %y to i64
  %call = call i32 (ptr, ...) @printf(ptr @.str, i64 %0)
  ret void
}

; test13a: Addr-of struct element, GEP followed by callinst.
;          no ssp attribute
; Requires no protector.
define void @test13a() {
entry:
; LINUX-I386-LABEL: test13a:
; LINUX-I386-NOT: calll __stack_chk_fail
; LINUX-I386: .cfi_endproc

; LINUX-X64-LABEL: test13a:
; LINUX-X64-NOT: callq __stack_chk_fail
; LINUX-X64: .cfi_endproc

; LINUX-KERNEL-X64-LABEL: test13a:
; LINUX-KERNEL-X64-NOT: callq __stack_chk_fail
; LINUX-KERNEL-X64: .cfi_endproc

; DARWIN-X64-LABEL: test13a:
; DARWIN-X64-NOT: callq ___stack_chk_fail
; DARWIN-X64: .cfi_endproc

; MSVC-I386-LABEL: test13a:
; MSVC-I386-NOT: calll @__security_check_cookie@4
; MSVC-I386: retl
  %c = alloca %struct.pair, align 4
  %y = getelementptr inbounds %struct.pair, ptr %c, i64 0, i32 1
  %call = call i32 (ptr, ...) @printf(ptr @.str, ptr %y)
  ret void
}

; test13b: Addr-of struct element, GEP followed by callinst.
;          ssp attribute
; Requires no protector.
; Function Attrs: ssp
define void @test13b() #0 {
entry:
; LINUX-I386-LABEL: test13b:
; LINUX-I386-NOT: calll __stack_chk_fail
; LINUX-I386: .cfi_endproc

; LINUX-X64-LABEL: test13b:
; LINUX-X64-NOT: callq __stack_chk_fail
; LINUX-X64: .cfi_endproc

; LINUX-KERNEL-X64-LABEL: test13b:
; LINUX-KERNEL-X64-NOT: callq __stack_chk_fail
; LINUX-KERNEL-X64: .cfi_endproc

; DARWIN-X64-LABEL: test13b:
; DARWIN-X64-NOT: callq ___stack_chk_fail
; DARWIN-X64: .cfi_endproc

; MSVC-I386-LABEL: test13b:
; MSVC-I386-NOT: calll @__security_check_cookie@4
; MSVC-I386: retl
  %c = alloca %struct.pair, align 4
  %y = getelementptr inbounds %struct.pair, ptr %c, i64 0, i32 1
  %call = call i32 (ptr, ...) @printf(ptr @.str, ptr %y)
  ret void
}

; test13c: Addr-of struct element, GEP followed by callinst.
;          sspstrong attribute
; Requires protector.
; Function Attrs: sspstrong 
define void @test13c() #1 {
entry:
; LINUX-I386-LABEL: test13c:
; LINUX-I386: mov{{l|q}} %gs:
; LINUX-I386: calll __stack_chk_fail

; LINUX-X64-LABEL: test13c:
; LINUX-X64: mov{{l|q}} %fs:
; LINUX-X64: callq __stack_chk_fail

; LINUX-KERNEL-X64-LABEL: test13c:
; LINUX-KERNEL-X64: mov{{l|q}} %gs:
; LINUX-KERNEL-X64: callq __stack_chk_fail

; DARWIN-X64-LABEL: test13c:
; DARWIN-X64: mov{{l|q}} ___stack_chk_guard
; DARWIN-X64: callq ___stack_chk_fail

; MSVC-I386-LABEL: test13c:
; MSVC-I386: movl ___security_cookie,
; MSVC-I386: calll @__security_check_cookie@4
  %c = alloca %struct.pair, align 4
  %y = getelementptr inbounds %struct.pair, ptr %c, i64 0, i32 1
  %call = call i32 (ptr, ...) @printf(ptr @.str, ptr %y)
  ret void
}

; test13d: Addr-of struct element, GEP followed by callinst.
;          sspreq attribute
; Requires protector.
; Function Attrs: sspreq 
define void @test13d() #2 {
entry:
; LINUX-I386-LABEL: test13d:
; LINUX-I386: mov{{l|q}} %gs:
; LINUX-I386: calll __stack_chk_fail

; LINUX-X64-LABEL: test13d:
; LINUX-X64: mov{{l|q}} %fs:
; LINUX-X64: callq __stack_chk_fail

; LINUX-KERNEL-X64-LABEL: test13d:
; LINUX-KERNEL-X64: mov{{l|q}} %gs:
; LINUX-KERNEL-X64: callq __stack_chk_fail

; DARWIN-X64-LABEL: test13d:
; DARWIN-X64: mov{{l|q}} ___stack_chk_guard
; DARWIN-X64: callq ___stack_chk_fail

; MSVC-I386-LABEL: test13d:
; MSVC-I386: movl ___security_cookie,
; MSVC-I386: calll @__security_check_cookie@4
  %c = alloca %struct.pair, align 4
  %y = getelementptr inbounds %struct.pair, ptr %c, i64 0, i32 1
  %call = call i32 (ptr, ...) @printf(ptr @.str, ptr %y)
  ret void
}

; test14a: Addr-of a local, optimized into a GEP (e.g., &a - 12)
;          no ssp attribute
; Requires no protector.
define void @test14a() {
entry:
; LINUX-I386-LABEL: test14a:
; LINUX-I386-NOT: calll __stack_chk_fail
; LINUX-I386: .cfi_endproc

; LINUX-X64-LABEL: test14a:
; LINUX-X64-NOT: callq __stack_chk_fail
; LINUX-X64: .cfi_endproc

; LINUX-KERNEL-X64-LABEL: test14a:
; LINUX-KERNEL-X64-NOT: callq __stack_chk_fail
; LINUX-KERNEL-X64: .cfi_endproc

; DARWIN-X64-LABEL: test14a:
; DARWIN-X64-NOT: callq ___stack_chk_fail
; DARWIN-X64: .cfi_endproc

; MSVC-I386-LABEL: test14a:
; MSVC-I386-NOT: calll @__security_check_cookie@4
; MSVC-I386: retl
  %a = alloca i32, align 4
  %add.ptr5 = getelementptr inbounds i32, ptr %a, i64 -12
  %call = call i32 (ptr, ...) @printf(ptr @.str, ptr %add.ptr5)
  ret void
}

; test14b: Addr-of a local, optimized into a GEP (e.g., &a - 12)
;          ssp attribute
; Requires no protector.
; Function Attrs: ssp
define void @test14b() #0 {
entry:
; LINUX-I386-LABEL: test14b:
; LINUX-I386-NOT: calll __stack_chk_fail
; LINUX-I386: .cfi_endproc

; LINUX-X64-LABEL: test14b:
; LINUX-X64-NOT: callq __stack_chk_fail
; LINUX-X64: .cfi_endproc

; LINUX-KERNEL-X64-LABEL: test14b:
; LINUX-KERNEL-X64-NOT: callq __stack_chk_fail
; LINUX-KERNEL-X64: .cfi_endproc

; DARWIN-X64-LABEL: test14b:
; DARWIN-X64-NOT: callq ___stack_chk_fail
; DARWIN-X64: .cfi_endproc

; MSVC-I386-LABEL: test14b:
; MSVC-I386-NOT: calll @__security_check_cookie@4
; MSVC-I386: retl
  %a = alloca i32, align 4
  %add.ptr5 = getelementptr inbounds i32, ptr %a, i64 -12
  %call = call i32 (ptr, ...) @printf(ptr @.str, ptr %add.ptr5)
  ret void
}

; test14c: Addr-of a local, optimized into a GEP (e.g., &a - 12)
;          sspstrong attribute
; Requires protector.
; Function Attrs: sspstrong 
define void @test14c() #1 {
entry:
; LINUX-I386-LABEL: test14c:
; LINUX-I386: mov{{l|q}} %gs:
; LINUX-I386: calll __stack_chk_fail

; LINUX-X64-LABEL: test14c:
; LINUX-X64: mov{{l|q}} %fs:
; LINUX-X64: callq __stack_chk_fail

; LINUX-KERNEL-X64-LABEL: test14c:
; LINUX-KERNEL-X64: mov{{l|q}} %gs:
; LINUX-KERNEL-X64: callq __stack_chk_fail

; DARWIN-X64-LABEL: test14c:
; DARWIN-X64: mov{{l|q}} ___stack_chk_guard
; DARWIN-X64: callq ___stack_chk_fail

; MSVC-I386-LABEL: test14c:
; MSVC-I386: movl ___security_cookie,
; MSVC-I386: calll @__security_check_cookie@4
  %a = alloca i32, align 4
  %add.ptr5 = getelementptr inbounds i32, ptr %a, i64 -12
  %call = call i32 (ptr, ...) @printf(ptr @.str, ptr %add.ptr5)
  ret void
}

; test14d: Addr-of a local, optimized into a GEP (e.g., &a - 12)
;          sspreq  attribute
; Requires protector.
; Function Attrs: sspreq 
define void @test14d() #2 {
entry:
; LINUX-I386-LABEL: test14d:
; LINUX-I386: mov{{l|q}} %gs:
; LINUX-I386: calll __stack_chk_fail

; LINUX-X64-LABEL: test14d:
; LINUX-X64: mov{{l|q}} %fs:
; LINUX-X64: callq __stack_chk_fail

; LINUX-KERNEL-X64-LABEL: test14d:
; LINUX-KERNEL-X64: mov{{l|q}} %gs:
; LINUX-KERNEL-X64: callq __stack_chk_fail

; DARWIN-X64-LABEL: test14d:
; DARWIN-X64: mov{{l|q}} ___stack_chk_guard
; DARWIN-X64: callq ___stack_chk_fail

; MSVC-I386-LABEL: test14d:
; MSVC-I386: movl ___security_cookie,
; MSVC-I386: calll @__security_check_cookie@4
  %a = alloca i32, align 4
  %add.ptr5 = getelementptr inbounds i32, ptr %a, i64 -12
  %call = call i32 (ptr, ...) @printf(ptr @.str, ptr %add.ptr5)
  ret void
}

; test15a: Addr-of a local cast to a ptr of a different type
;           (e.g., int a; ... ; ptr b = &a;)
;          no ssp attribute
; Requires no protector.
define void @test15a() {
entry:
; LINUX-I386-LABEL: test15a:
; LINUX-I386-NOT: calll __stack_chk_fail
; LINUX-I386: .cfi_endproc

; LINUX-X64-LABEL: test15a:
; LINUX-X64-NOT: callq __stack_chk_fail
; LINUX-X64: .cfi_endproc

; LINUX-KERNEL-X64-LABEL: test15a:
; LINUX-KERNEL-X64-NOT: callq __stack_chk_fail
; LINUX-KERNEL-X64: .cfi_endproc

; DARWIN-X64-LABEL: test15a:
; DARWIN-X64-NOT: callq ___stack_chk_fail
; DARWIN-X64: .cfi_endproc

; MSVC-I386-LABEL: test15a:
; MSVC-I386-NOT: calll @__security_check_cookie@4
; MSVC-I386: retl
  %a = alloca i32, align 4
  %b = alloca ptr, align 8
  store i32 0, ptr %a, align 4
  store ptr %a, ptr %b, align 8
  %0 = load ptr, ptr %b, align 8
  %call = call i32 (ptr, ...) @printf(ptr @.str, ptr %0)
  ret void
}

; test15b: Addr-of a local cast to a ptr of a different type
;           (e.g., int a; ... ; ptr b = &a;)
;          ssp attribute
; Requires no protector.
; Function Attrs: ssp
define void @test15b() #0 {
entry:
; LINUX-I386-LABEL: test15b:
; LINUX-I386-NOT: calll __stack_chk_fail
; LINUX-I386: .cfi_endproc

; LINUX-X64-LABEL: test15b:
; LINUX-X64-NOT: callq __stack_chk_fail
; LINUX-X64: .cfi_endproc

; LINUX-KERNEL-X64-LABEL: test15b:
; LINUX-KERNEL-X64-NOT: callq __stack_chk_fail
; LINUX-KERNEL-X64: .cfi_endproc

; DARWIN-X64-LABEL: test15b:
; DARWIN-X64-NOT: callq ___stack_chk_fail
; DARWIN-X64: .cfi_endproc

; MSVC-I386-LABEL: test15b:
; MSVC-I386-NOT: calll @__security_check_cookie@4
; MSVC-I386: retl
  %a = alloca i32, align 4
  %b = alloca ptr, align 8
  store i32 0, ptr %a, align 4
  store ptr %a, ptr %b, align 8
  %0 = load ptr, ptr %b, align 8
  %call = call i32 (ptr, ...) @printf(ptr @.str, ptr %0)
  ret void
}

; test15c: Addr-of a local cast to a ptr of a different type
;           (e.g., int a; ... ; ptr b = &a;)
;          sspstrong attribute
; Requires protector.
; Function Attrs: sspstrong 
define void @test15c() #1 {
entry:
; LINUX-I386-LABEL: test15c:
; LINUX-I386: mov{{l|q}} %gs:
; LINUX-I386: calll __stack_chk_fail

; LINUX-X64-LABEL: test15c:
; LINUX-X64: mov{{l|q}} %fs:
; LINUX-X64: callq __stack_chk_fail

; LINUX-KERNEL-X64-LABEL: test15c:
; LINUX-KERNEL-X64: mov{{l|q}} %gs:
; LINUX-KERNEL-X64: callq __stack_chk_fail

; DARWIN-X64-LABEL: test15c:
; DARWIN-X64: mov{{l|q}} ___stack_chk_guard
; DARWIN-X64: callq ___stack_chk_fail

; MSVC-I386-LABEL: test15c:
; MSVC-I386: movl ___security_cookie,
; MSVC-I386: calll @__security_check_cookie@4
  %a = alloca i32, align 4
  %b = alloca ptr, align 8
  store i32 0, ptr %a, align 4
  store ptr %a, ptr %b, align 8
  %0 = load ptr, ptr %b, align 8
  %call = call i32 (ptr, ...) @printf(ptr @.str, ptr %0)
  ret void
}

; test15d: Addr-of a local cast to a ptr of a different type
;           (e.g., int a; ... ; ptr b = &a;)
;          sspreq attribute
; Requires protector.
; Function Attrs: sspreq 
define void @test15d() #2 {
entry:
; LINUX-I386-LABEL: test15d:
; LINUX-I386: mov{{l|q}} %gs:
; LINUX-I386: calll __stack_chk_fail

; LINUX-X64-LABEL: test15d:
; LINUX-X64: mov{{l|q}} %fs:
; LINUX-X64: callq __stack_chk_fail

; LINUX-KERNEL-X64-LABEL: test15d:
; LINUX-KERNEL-X64: mov{{l|q}} %gs:
; LINUX-KERNEL-X64: callq __stack_chk_fail

; DARWIN-X64-LABEL: test15d:
; DARWIN-X64: mov{{l|q}} ___stack_chk_guard
; DARWIN-X64: callq ___stack_chk_fail

; MSVC-I386-LABEL: test15d:
; MSVC-I386: movl ___security_cookie,
; MSVC-I386: calll @__security_check_cookie@4
  %a = alloca i32, align 4
  %b = alloca ptr, align 8
  store i32 0, ptr %a, align 4
  store ptr %a, ptr %b, align 8
  %0 = load ptr, ptr %b, align 8
  %call = call i32 (ptr, ...) @printf(ptr @.str, ptr %0)
  ret void
}

; test16a: Addr-of a local cast to a ptr of a different type (optimized)
;           (e.g., int a; ... ; ptr b = &a;)
;          no ssp attribute
; Requires no protector.
define void @test16a() {
entry:
; LINUX-I386-LABEL: test16a:
; LINUX-I386-NOT: calll __stack_chk_fail
; LINUX-I386: .cfi_endproc

; LINUX-X64-LABEL: test16a:
; LINUX-X64-NOT: callq __stack_chk_fail
; LINUX-X64: .cfi_endproc

; LINUX-KERNEL-X64-LABEL: test16a:
; LINUX-KERNEL-X64-NOT: callq __stack_chk_fail
; LINUX-KERNEL-X64: .cfi_endproc

; DARWIN-X64-LABEL: test16a:
; DARWIN-X64-NOT: callq ___stack_chk_fail
; DARWIN-X64: .cfi_endproc

; MSVC-I386-LABEL: test16a:
; MSVC-I386-NOT: calll @__security_check_cookie@4
; MSVC-I386: retl
  %a = alloca i32, align 4
  store i32 0, ptr %a, align 4
  call void @funfloat(ptr %a)
  ret void
}

; test16b: Addr-of a local cast to a ptr of a different type (optimized)
;           (e.g., int a; ... ; ptr b = &a;)
;          ssp attribute
; Requires no protector.
; Function Attrs: ssp
define void @test16b() #0 {
entry:
; LINUX-I386-LABEL: test16b:
; LINUX-I386-NOT: calll __stack_chk_fail
; LINUX-I386: .cfi_endproc

; LINUX-X64-LABEL: test16b:
; LINUX-X64-NOT: callq __stack_chk_fail
; LINUX-X64: .cfi_endproc

; LINUX-KERNEL-X64-LABEL: test16b:
; LINUX-KERNEL-X64-NOT: callq __stack_chk_fail
; LINUX-KERNEL-X64: .cfi_endproc

; DARWIN-X64-LABEL: test16b:
; DARWIN-X64-NOT: callq ___stack_chk_fail
; DARWIN-X64: .cfi_endproc

; MSVC-I386-LABEL: test16b:
; MSVC-I386-NOT: calll @__security_check_cookie@4
; MSVC-I386: retl
  %a = alloca i32, align 4
  store i32 0, ptr %a, align 4
  call void @funfloat(ptr %a)
  ret void
}

; test16c: Addr-of a local cast to a ptr of a different type (optimized)
;           (e.g., int a; ... ; ptr b = &a;)
;          sspstrong attribute
; Requires protector.
; Function Attrs: sspstrong 
define void @test16c() #1 {
entry:
; LINUX-I386-LABEL: test16c:
; LINUX-I386: mov{{l|q}} %gs:
; LINUX-I386: calll __stack_chk_fail

; LINUX-X64-LABEL: test16c:
; LINUX-X64: mov{{l|q}} %fs:
; LINUX-X64: callq __stack_chk_fail

; LINUX-KERNEL-X64-LABEL: test16c:
; LINUX-KERNEL-X64: mov{{l|q}} %gs:
; LINUX-KERNEL-X64: callq __stack_chk_fail

; DARWIN-X64-LABEL: test16c:
; DARWIN-X64: mov{{l|q}} ___stack_chk_guard
; DARWIN-X64: callq ___stack_chk_fail

; MSVC-I386-LABEL: test16c:
; MSVC-I386: movl ___security_cookie,
; MSVC-I386: calll @__security_check_cookie@4
  %a = alloca i32, align 4
  store i32 0, ptr %a, align 4
  call void @funfloat(ptr %a)
  ret void
}

; test16d: Addr-of a local cast to a ptr of a different type (optimized)
;           (e.g., int a; ... ; ptr b = &a;)
;          sspreq attribute
; Requires protector.
; Function Attrs: sspreq 
define void @test16d() #2 {
entry:
; LINUX-I386-LABEL: test16d:
; LINUX-I386: mov{{l|q}} %gs:
; LINUX-I386: calll __stack_chk_fail

; LINUX-X64-LABEL: test16d:
; LINUX-X64: mov{{l|q}} %fs:
; LINUX-X64: callq __stack_chk_fail

; LINUX-KERNEL-X64-LABEL: test16d:
; LINUX-KERNEL-X64: mov{{l|q}} %gs:
; LINUX-KERNEL-X64: callq __stack_chk_fail

; DARWIN-X64-LABEL: test16d:
; DARWIN-X64: mov{{l|q}} ___stack_chk_guard
; DARWIN-X64: callq ___stack_chk_fail

; MSVC-I386-LABEL: test16d:
; MSVC-I386: movl ___security_cookie,
; MSVC-I386: calll @__security_check_cookie@4
  %a = alloca i32, align 4
  store i32 0, ptr %a, align 4
  call void @funfloat(ptr %a)
  ret void
}

; test17a: Addr-of a vector nested in a struct
;          no ssp attribute
; Requires no protector.
define void @test17a() {
entry:
; LINUX-I386-LABEL: test17a:
; LINUX-I386-NOT: calll __stack_chk_fail
; LINUX-I386: .cfi_endproc

; LINUX-X64-LABEL: test17a:
; LINUX-X64-NOT: callq __stack_chk_fail
; LINUX-X64: .cfi_endproc

; LINUX-KERNEL-X64-LABEL: test17a:
; LINUX-KERNEL-X64-NOT: callq __stack_chk_fail
; LINUX-KERNEL-X64: .cfi_endproc

; DARWIN-X64-LABEL: test17a:
; DARWIN-X64-NOT: callq ___stack_chk_fail
; DARWIN-X64: .cfi_endproc

; MSVC-I386-LABEL: test17a:
; MSVC-I386-NOT: calll @__security_check_cookie@4
; MSVC-I386: retl
  %c = alloca %struct.vec, align 16
  %add.ptr = getelementptr inbounds <4 x i32>, ptr %c, i64 -12
  %call = call i32 (ptr, ...) @printf(ptr @.str, ptr %add.ptr)
  ret void
}

; test17b: Addr-of a vector nested in a struct
;          ssp attribute
; Requires no protector.
; Function Attrs: ssp
define void @test17b() #0 {
entry:
; LINUX-I386-LABEL: test17b:
; LINUX-I386-NOT: calll __stack_chk_fail
; LINUX-I386: .cfi_endproc

; LINUX-X64-LABEL: test17b:
; LINUX-X64-NOT: callq __stack_chk_fail
; LINUX-X64: .cfi_endproc

; LINUX-KERNEL-X64-LABEL: test17b:
; LINUX-KERNEL-X64-NOT: callq __stack_chk_fail
; LINUX-KERNEL-X64: .cfi_endproc

; DARWIN-X64-LABEL: test17b:
; DARWIN-X64-NOT: callq ___stack_chk_fail
; DARWIN-X64: .cfi_endproc

; MSVC-I386-LABEL: test17b:
; MSVC-I386-NOT: calll @__security_check_cookie@4
; MSVC-I386: retl
  %c = alloca %struct.vec, align 16
  %add.ptr = getelementptr inbounds <4 x i32>, ptr %c, i64 -12
  %call = call i32 (ptr, ...) @printf(ptr @.str, ptr %add.ptr)
  ret void
}

; test17c: Addr-of a vector nested in a struct
;          sspstrong attribute
; Requires protector.
; Function Attrs: sspstrong 
define void @test17c() #1 {
entry:
; LINUX-I386-LABEL: test17c:
; LINUX-I386: mov{{l|q}} %gs:
; LINUX-I386: calll __stack_chk_fail

; LINUX-X64-LABEL: test17c:
; LINUX-X64: mov{{l|q}} %fs:
; LINUX-X64: callq __stack_chk_fail

; LINUX-KERNEL-X64-LABEL: test17c:
; LINUX-KERNEL-X64: mov{{l|q}} %gs:
; LINUX-KERNEL-X64: callq __stack_chk_fail

; DARWIN-X64-LABEL: test17c:
; DARWIN-X64: mov{{l|q}} ___stack_chk_guard
; DARWIN-X64: callq ___stack_chk_fail

; MSVC-I386-LABEL: test17c:
; MSVC-I386: movl ___security_cookie,
; MSVC-I386: calll @__security_check_cookie@4
  %c = alloca %struct.vec, align 16
  %add.ptr = getelementptr inbounds <4 x i32>, ptr %c, i64 -12
  %call = call i32 (ptr, ...) @printf(ptr @.str, ptr %add.ptr)
  ret void
}

; test17d: Addr-of a vector nested in a struct
;          sspreq attribute
; Requires protector.
; Function Attrs: sspreq 
define void @test17d() #2 {
entry:
; LINUX-I386-LABEL: test17d:
; LINUX-I386: mov{{l|q}} %gs:
; LINUX-I386: calll __stack_chk_fail

; LINUX-X64-LABEL: test17d:
; LINUX-X64: mov{{l|q}} %fs:
; LINUX-X64: callq __stack_chk_fail

; LINUX-KERNEL-X64-LABEL: test17d:
; LINUX-KERNEL-X64: mov{{l|q}} %gs:
; LINUX-KERNEL-X64: callq __stack_chk_fail

; DARWIN-X64-LABEL: test17d:
; DARWIN-X64: mov{{l|q}} ___stack_chk_guard
; DARWIN-X64: callq ___stack_chk_fail

; MSVC-I386-LABEL: test17d:
; MSVC-I386: movl ___security_cookie,
; MSVC-I386: calll @__security_check_cookie@4
  %c = alloca %struct.vec, align 16
  %add.ptr = getelementptr inbounds <4 x i32>, ptr %c, i64 -12
  %call = call i32 (ptr, ...) @printf(ptr @.str, ptr %add.ptr)
  ret void
}

; test18a: Addr-of a variable passed into an invoke instruction.
;          no ssp attribute
; Requires no protector.
define i32 @test18a() personality ptr @__gxx_personality_v0 {
entry:
; LINUX-I386-LABEL: test18a:
; LINUX-I386-NOT: calll __stack_chk_fail
; LINUX-I386: .cfi_endproc

; LINUX-X64-LABEL: test18a:
; LINUX-X64-NOT: callq __stack_chk_fail
; LINUX-X64: .cfi_endproc

; LINUX-KERNEL-X64-LABEL: test18a:
; LINUX-KERNEL-X64-NOT: callq __stack_chk_fail
; LINUX-KERNEL-X64: .cfi_endproc

; DARWIN-X64-LABEL: test18a:
; DARWIN-X64-NOT: callq ___stack_chk_fail
; DARWIN-X64: .cfi_endproc

; MSVC-I386-LABEL: test18a:
; MSVC-I386-NOT: calll @__security_check_cookie@4
; MSVC-I386: retl
  %a = alloca i32, align 4
  %exn.slot = alloca ptr
  %ehselector.slot = alloca i32
  store i32 0, ptr %a, align 4
  invoke void @_Z3exceptPi(ptr %a)
          to label %invoke.cont unwind label %lpad

invoke.cont:
  ret i32 0

lpad:
  %0 = landingpad { ptr, i32 }
          catch ptr null
  ret i32 0
}

; test18b: Addr-of a variable passed into an invoke instruction.
;          ssp attribute
; Requires no protector.
; Function Attrs: ssp 
define i32 @test18b() #0 personality ptr @__gxx_personality_v0 {
entry:
; LINUX-I386-LABEL: test18b:
; LINUX-I386-NOT: calll __stack_chk_fail
; LINUX-I386: .cfi_endproc

; LINUX-X64-LABEL: test18b:
; LINUX-X64-NOT: callq __stack_chk_fail
; LINUX-X64: .cfi_endproc

; LINUX-KERNEL-X64-LABEL: test18b:
; LINUX-KERNEL-X64-NOT: callq __stack_chk_fail
; LINUX-KERNEL-X64: .cfi_endproc

; DARWIN-X64-LABEL: test18b:
; DARWIN-X64-NOT: callq ___stack_chk_fail
; DARWIN-X64: .cfi_endproc

; MSVC-I386-LABEL: test18b:
; MSVC-I386-NOT: calll @__security_check_cookie@4
; MSVC-I386: retl
  %a = alloca i32, align 4
  %exn.slot = alloca ptr
  %ehselector.slot = alloca i32
  store i32 0, ptr %a, align 4
  invoke void @_Z3exceptPi(ptr %a)
          to label %invoke.cont unwind label %lpad

invoke.cont:
  ret i32 0

lpad:
  %0 = landingpad { ptr, i32 }
          catch ptr null
  ret i32 0
}

; test18c: Addr-of a variable passed into an invoke instruction.
;          sspstrong attribute
; Requires protector.
; Function Attrs: sspstrong 
define i32 @test18c() #1 personality ptr @__gxx_personality_v0 {
entry:
; LINUX-I386-LABEL: test18c:
; LINUX-I386: mov{{l|q}} %gs:
; LINUX-I386: calll __stack_chk_fail

; LINUX-X64-LABEL: test18c:
; LINUX-X64: mov{{l|q}} %fs:
; LINUX-X64: callq __stack_chk_fail

; LINUX-KERNEL-X64-LABEL: test18c:
; LINUX-KERNEL-X64: mov{{l|q}} %gs:
; LINUX-KERNEL-X64: callq __stack_chk_fail

; DARWIN-X64-LABEL: test18c:
; DARWIN-X64: mov{{l|q}} ___stack_chk_guard
; DARWIN-X64: callq ___stack_chk_fail

; MSVC-I386-LABEL: test18c:
; MSVC-I386: movl ___security_cookie,
; MSVC-I386: calll @__security_check_cookie@4
  %a = alloca i32, align 4
  %exn.slot = alloca ptr
  %ehselector.slot = alloca i32
  store i32 0, ptr %a, align 4
  invoke void @_Z3exceptPi(ptr %a)
          to label %invoke.cont unwind label %lpad

invoke.cont:
  ret i32 0

lpad:
  %0 = landingpad { ptr, i32 }
          catch ptr null
  ret i32 0
}

; test18d: Addr-of a variable passed into an invoke instruction.
;          sspreq attribute
; Requires protector.
; Function Attrs: sspreq 
define i32 @test18d() #2 personality ptr @__gxx_personality_v0 {
entry:
; LINUX-I386-LABEL: test18d:
; LINUX-I386: mov{{l|q}} %gs:
; LINUX-I386: calll __stack_chk_fail

; LINUX-X64-LABEL: test18d:
; LINUX-X64: mov{{l|q}} %fs:
; LINUX-X64: callq __stack_chk_fail

; LINUX-KERNEL-X64-LABEL: test18d:
; LINUX-KERNEL-X64: mov{{l|q}} %gs:
; LINUX-KERNEL-X64: callq __stack_chk_fail

; DARWIN-X64-LABEL: test18d:
; DARWIN-X64: mov{{l|q}} ___stack_chk_guard
; DARWIN-X64: callq ___stack_chk_fail

; MSVC-I386-LABEL: test18d:
; MSVC-I386: movl ___security_cookie,
; MSVC-I386: calll @__security_check_cookie@4
  %a = alloca i32, align 4
  %exn.slot = alloca ptr
  %ehselector.slot = alloca i32
  store i32 0, ptr %a, align 4
  invoke void @_Z3exceptPi(ptr %a)
          to label %invoke.cont unwind label %lpad

invoke.cont:
  ret i32 0

lpad:
  %0 = landingpad { ptr, i32 }
          catch ptr null
  ret i32 0
}
; test19a: Addr-of a struct element passed into an invoke instruction.
;           (GEP followed by an invoke)
;          no ssp attribute
; Requires no protector.
define i32 @test19a() personality ptr @__gxx_personality_v0 {
entry:
; LINUX-I386-LABEL: test19a:
; LINUX-I386-NOT: calll __stack_chk_fail
; LINUX-I386: .cfi_endproc

; LINUX-X64-LABEL: test19a:
; LINUX-X64-NOT: callq __stack_chk_fail
; LINUX-X64: .cfi_endproc

; LINUX-KERNEL-X64-LABEL: test19a:
; LINUX-KERNEL-X64-NOT: callq __stack_chk_fail
; LINUX-KERNEL-X64: .cfi_endproc

; DARWIN-X64-LABEL: test19a:
; DARWIN-X64-NOT: callq ___stack_chk_fail
; DARWIN-X64: .cfi_endproc

; MSVC-I386-LABEL: test19a:
; MSVC-I386-NOT: calll @__security_check_cookie@4
; MSVC-I386: retl
  %c = alloca %struct.pair, align 4
  %exn.slot = alloca ptr
  %ehselector.slot = alloca i32
  store i32 0, ptr %c, align 4
  invoke void @_Z3exceptPi(ptr %c)
          to label %invoke.cont unwind label %lpad

invoke.cont:
  ret i32 0

lpad:
  %0 = landingpad { ptr, i32 }
          catch ptr null
  ret i32 0
}

; test19b: Addr-of a struct element passed into an invoke instruction.
;           (GEP followed by an invoke)
;          ssp attribute
; Requires no protector.
; Function Attrs: ssp 
define i32 @test19b() #0 personality ptr @__gxx_personality_v0 {
entry:
; LINUX-I386-LABEL: test19b:
; LINUX-I386-NOT: calll __stack_chk_fail
; LINUX-I386: .cfi_endproc

; LINUX-X64-LABEL: test19b:
; LINUX-X64-NOT: callq __stack_chk_fail
; LINUX-X64: .cfi_endproc

; LINUX-KERNEL-X64-LABEL: test19b:
; LINUX-KERNEL-X64-NOT: callq __stack_chk_fail
; LINUX-KERNEL-X64: .cfi_endproc

; DARWIN-X64-LABEL: test19b:
; DARWIN-X64-NOT: callq ___stack_chk_fail
; DARWIN-X64: .cfi_endproc

; MSVC-I386-LABEL: test19b:
; MSVC-I386-NOT: calll @__security_check_cookie@4
; MSVC-I386: retl
  %c = alloca %struct.pair, align 4
  %exn.slot = alloca ptr
  %ehselector.slot = alloca i32
  store i32 0, ptr %c, align 4
  invoke void @_Z3exceptPi(ptr %c)
          to label %invoke.cont unwind label %lpad

invoke.cont:
  ret i32 0

lpad:
  %0 = landingpad { ptr, i32 }
          catch ptr null
  ret i32 0
}

; test19c: Addr-of a struct element passed into an invoke instruction.
;           (GEP followed by an invoke)
;          sspstrong attribute
; Requires protector.
; Function Attrs: sspstrong 
define i32 @test19c() #1 personality ptr @__gxx_personality_v0 {
entry:
; LINUX-I386-LABEL: test19c:
; LINUX-I386: mov{{l|q}} %gs:
; LINUX-I386: calll __stack_chk_fail

; LINUX-X64-LABEL: test19c:
; LINUX-X64: mov{{l|q}} %fs:
; LINUX-X64: callq __stack_chk_fail

; LINUX-KERNEL-X64-LABEL: test19c:
; LINUX-KERNEL-X64: mov{{l|q}} %gs:
; LINUX-KERNEL-X64: callq __stack_chk_fail

; DARWIN-X64-LABEL: test19c:
; DARWIN-X64: mov{{l|q}} ___stack_chk_guard
; DARWIN-X64: callq ___stack_chk_fail

; MSVC-I386-LABEL: test19c:
; MSVC-I386: movl ___security_cookie,
; MSVC-I386: calll @__security_check_cookie@4
  %c = alloca %struct.pair, align 4
  %exn.slot = alloca ptr
  %ehselector.slot = alloca i32
  store i32 0, ptr %c, align 4
  invoke void @_Z3exceptPi(ptr %c)
          to label %invoke.cont unwind label %lpad

invoke.cont:
  ret i32 0

lpad:
  %0 = landingpad { ptr, i32 }
          catch ptr null
  ret i32 0
}

; test19d: Addr-of a struct element passed into an invoke instruction.
;           (GEP followed by an invoke)
;          sspreq attribute
; Requires protector.
; Function Attrs: sspreq 
define i32 @test19d() #2 personality ptr @__gxx_personality_v0 {
entry:
; LINUX-I386-LABEL: test19d:
; LINUX-I386: mov{{l|q}} %gs:
; LINUX-I386: calll __stack_chk_fail
; LINUX-I386-NOT: calll __stack_chk_fail

; LINUX-X64-LABEL: test19d:
; LINUX-X64: mov{{l|q}} %fs:
; LINUX-X64: callq __stack_chk_fail
; LINUX-X64-NOT: callq __stack_chk_fail

; LINUX-KERNEL-X64-LABEL: test19d:
; LINUX-KERNEL-X64: mov{{l|q}} %gs:
; LINUX-KERNEL-X64: callq __stack_chk_fail
; LINUX-KERNEL-X64-NOT: callq ___stack_chk_fail

; DARWIN-X64-LABEL: test19d:
; DARWIN-X64: mov{{l|q}} ___stack_chk_guard
; DARWIN-X64: callq ___stack_chk_fail
; DARWIN-X64-NOT: callq ___stack_chk_fail

; MSVC-I386-LABEL: test19d:
; MSVC-I386: movl ___security_cookie,
; MSVC-I386: calll @__security_check_cookie@4

; MINGW-X64-LABEL: test19d:
; MINGW-X64: mov{{l|q}} .refptr.__stack_chk_guard
; MINGW-X64: callq __stack_chk_fail

  %c = alloca %struct.pair, align 4
  %exn.slot = alloca ptr
  %ehselector.slot = alloca i32
  store i32 0, ptr %c, align 4
  invoke void @_Z3exceptPi(ptr %c)
          to label %invoke.cont unwind label %lpad

invoke.cont:
  ret i32 0

lpad:
  %0 = landingpad { ptr, i32 }
          catch ptr null
  ret i32 0
}

; test20a: Addr-of a pointer
;          no ssp attribute
; Requires no protector.
define void @test20a() {
entry:
; LINUX-I386-LABEL: test20a:
; LINUX-I386-NOT: calll __stack_chk_fail
; LINUX-I386: .cfi_endproc

; LINUX-X64-LABEL: test20a:
; LINUX-X64-NOT: callq __stack_chk_fail
; LINUX-X64: .cfi_endproc

; LINUX-KERNEL-X64-LABEL: test20a:
; LINUX-KERNEL-X64-NOT: callq __stack_chk_fail
; LINUX-KERNEL-X64: .cfi_endproc

; DARWIN-X64-LABEL: test20a:
; DARWIN-X64-NOT: callq ___stack_chk_fail
; DARWIN-X64: .cfi_endproc

; MSVC-I386-LABEL: test20a:
; MSVC-I386-NOT: calll @__security_check_cookie@4
; MSVC-I386: retl
  %a = alloca ptr, align 8
  %b = alloca ptr, align 8
  %call = call ptr @getp()
  store ptr %call, ptr %a, align 8
  store ptr %a, ptr %b, align 8
  %0 = load ptr, ptr %b, align 8
  call void @funcall2(ptr %0)
  ret void
}

; test20b: Addr-of a pointer
;          ssp attribute
; Requires no protector.
; Function Attrs: ssp
define void @test20b() #0 {
entry:
; LINUX-I386-LABEL: test20b:
; LINUX-I386-NOT: calll __stack_chk_fail
; LINUX-I386: .cfi_endproc

; LINUX-X64-LABEL: test20b:
; LINUX-X64-NOT: callq __stack_chk_fail
; LINUX-X64: .cfi_endproc

; LINUX-KERNEL-X64-LABEL: test20b:
; LINUX-KERNEL-X64-NOT: callq __stack_chk_fail
; LINUX-KERNEL-X64: .cfi_endproc

; DARWIN-X64-LABEL: test20b:
; DARWIN-X64-NOT: callq ___stack_chk_fail
; DARWIN-X64: .cfi_endproc

; MSVC-I386-LABEL: test20b:
; MSVC-I386-NOT: calll @__security_check_cookie@4
; MSVC-I386: retl
  %a = alloca ptr, align 8
  %b = alloca ptr, align 8
  %call = call ptr @getp()
  store ptr %call, ptr %a, align 8
  store ptr %a, ptr %b, align 8
  %0 = load ptr, ptr %b, align 8
  call void @funcall2(ptr %0)
  ret void
}

; test20c: Addr-of a pointer
;          sspstrong attribute
; Requires protector.
; Function Attrs: sspstrong 
define void @test20c() #1 {
entry:
; LINUX-I386-LABEL: test20c:
; LINUX-I386: mov{{l|q}} %gs:
; LINUX-I386: calll __stack_chk_fail

; LINUX-X64-LABEL: test20c:
; LINUX-X64: mov{{l|q}} %fs:
; LINUX-X64: callq __stack_chk_fail

; LINUX-KERNEL-X64-LABEL: test20c:
; LINUX-KERNEL-X64: mov{{l|q}} %gs:
; LINUX-KERNEL-X64: callq __stack_chk_fail

; DARWIN-X64-LABEL: test20c:
; DARWIN-X64: mov{{l|q}} ___stack_chk_guard
; DARWIN-X64: callq ___stack_chk_fail

; MSVC-I386-LABEL: test20c:
; MSVC-I386: movl ___security_cookie,
; MSVC-I386: calll @__security_check_cookie@4
  %a = alloca ptr, align 8
  %b = alloca ptr, align 8
  %call = call ptr @getp()
  store ptr %call, ptr %a, align 8
  store ptr %a, ptr %b, align 8
  %0 = load ptr, ptr %b, align 8
  call void @funcall2(ptr %0)
  ret void
}

; test20d: Addr-of a pointer
;          sspreq attribute
; Requires protector.
; Function Attrs: sspreq 
define void @test20d() #2 {
entry:
; LINUX-I386-LABEL: test20d:
; LINUX-I386: mov{{l|q}} %gs:
; LINUX-I386: calll __stack_chk_fail

; LINUX-X64-LABEL: test20d:
; LINUX-X64: mov{{l|q}} %fs:
; LINUX-X64: callq __stack_chk_fail

; LINUX-KERNEL-X64-LABEL: test20d:
; LINUX-KERNEL-X64: mov{{l|q}} %gs:
; LINUX-KERNEL-X64: callq __stack_chk_fail

; DARWIN-X64-LABEL: test20d:
; DARWIN-X64: mov{{l|q}} ___stack_chk_guard
; DARWIN-X64: callq ___stack_chk_fail

; MSVC-I386-LABEL: test20d:
; MSVC-I386: movl ___security_cookie,
; MSVC-I386: calll @__security_check_cookie@4
  %a = alloca ptr, align 8
  %b = alloca ptr, align 8
  %call = call ptr @getp()
  store ptr %call, ptr %a, align 8
  store ptr %a, ptr %b, align 8
  %0 = load ptr, ptr %b, align 8
  call void @funcall2(ptr %0)
  ret void
}

; test21a: Addr-of a casted pointer
;          no ssp attribute
; Requires no protector.
define void @test21a() {
entry:
; LINUX-I386-LABEL: test21a:
; LINUX-I386-NOT: calll __stack_chk_fail
; LINUX-I386: .cfi_endproc

; LINUX-X64-LABEL: test21a:
; LINUX-X64-NOT: callq __stack_chk_fail
; LINUX-X64: .cfi_endproc

; LINUX-KERNEL-X64-LABEL: test21a:
; LINUX-KERNEL-X64-NOT: callq __stack_chk_fail
; LINUX-KERNEL-X64: .cfi_endproc

; DARWIN-X64-LABEL: test21a:
; DARWIN-X64-NOT: callq ___stack_chk_fail
; DARWIN-X64: .cfi_endproc

; MSVC-I386-LABEL: test21a:
; MSVC-I386-NOT: calll @__security_check_cookie@4
; MSVC-I386: retl
  %a = alloca ptr, align 8
  %b = alloca ptr, align 8
  %call = call ptr @getp()
  store ptr %call, ptr %a, align 8
  store ptr %a, ptr %b, align 8
  %0 = load ptr, ptr %b, align 8
  call void @funfloat2(ptr %0)
  ret void
}

; test21b: Addr-of a casted pointer
;          ssp attribute
; Requires no protector.
; Function Attrs: ssp
define void @test21b() #0 {
entry:
; LINUX-I386-LABEL: test21b:
; LINUX-I386-NOT: calll __stack_chk_fail
; LINUX-I386: .cfi_endproc

; LINUX-X64-LABEL: test21b:
; LINUX-X64-NOT: callq __stack_chk_fail
; LINUX-X64: .cfi_endproc

; LINUX-KERNEL-X64-LABEL: test21b:
; LINUX-KERNEL-X64-NOT: callq __stack_chk_fail
; LINUX-KERNEL-X64: .cfi_endproc

; DARWIN-X64-LABEL: test21b:
; DARWIN-X64-NOT: callq ___stack_chk_fail
; DARWIN-X64: .cfi_endproc

; MSVC-I386-LABEL: test21b:
; MSVC-I386-NOT: calll @__security_check_cookie@4
; MSVC-I386: retl
  %a = alloca ptr, align 8
  %b = alloca ptr, align 8
  %call = call ptr @getp()
  store ptr %call, ptr %a, align 8
  store ptr %a, ptr %b, align 8
  %0 = load ptr, ptr %b, align 8
  call void @funfloat2(ptr %0)
  ret void
}

; test21c: Addr-of a casted pointer
;          sspstrong attribute
; Requires protector.
; Function Attrs: sspstrong 
define void @test21c() #1 {
entry:
; LINUX-I386-LABEL: test21c:
; LINUX-I386: mov{{l|q}} %gs:
; LINUX-I386: calll __stack_chk_fail

; LINUX-X64-LABEL: test21c:
; LINUX-X64: mov{{l|q}} %fs:
; LINUX-X64: callq __stack_chk_fail

; LINUX-KERNEL-X64-LABEL: test21c:
; LINUX-KERNEL-X64: mov{{l|q}} %gs:
; LINUX-KERNEL-X64: callq __stack_chk_fail

; DARWIN-X64-LABEL: test21c:
; DARWIN-X64: mov{{l|q}} ___stack_chk_guard
; DARWIN-X64: callq ___stack_chk_fail

; MSVC-I386-LABEL: test21c:
; MSVC-I386: movl ___security_cookie,
; MSVC-I386: calll @__security_check_cookie@4
  %a = alloca ptr, align 8
  %b = alloca ptr, align 8
  %call = call ptr @getp()
  store ptr %call, ptr %a, align 8
  store ptr %a, ptr %b, align 8
  %0 = load ptr, ptr %b, align 8
  call void @funfloat2(ptr %0)
  ret void
}

; test21d: Addr-of a casted pointer
;          sspreq attribute
; Requires protector.
; Function Attrs: sspreq 
define void @test21d() #2 {
entry:
; LINUX-I386-LABEL: test21d:
; LINUX-I386: mov{{l|q}} %gs:
; LINUX-I386: calll __stack_chk_fail

; LINUX-X64-LABEL: test21d:
; LINUX-X64: mov{{l|q}} %fs:
; LINUX-X64: callq __stack_chk_fail

; LINUX-KERNEL-X64-LABEL: test21d:
; LINUX-KERNEL-X64: mov{{l|q}} %gs:
; LINUX-KERNEL-X64: callq __stack_chk_fail

; DARWIN-X64-LABEL: test21d:
; DARWIN-X64: mov{{l|q}} ___stack_chk_guard
; DARWIN-X64: callq ___stack_chk_fail

; MSVC-I386-LABEL: test21d:
; MSVC-I386: movl ___security_cookie,
; MSVC-I386: calll @__security_check_cookie@4
  %a = alloca ptr, align 8
  %b = alloca ptr, align 8
  %call = call ptr @getp()
  store ptr %call, ptr %a, align 8
  store ptr %a, ptr %b, align 8
  %0 = load ptr, ptr %b, align 8
  call void @funfloat2(ptr %0)
  ret void
}

; test22a: [2 x i8] in a class
;          no ssp attribute
; Requires no protector.
define signext i8 @test22a() {
entry:
; LINUX-I386-LABEL: test22a:
; LINUX-I386-NOT: calll __stack_chk_fail
; LINUX-I386: .cfi_endproc

; LINUX-X64-LABEL: test22a:
; LINUX-X64-NOT: callq __stack_chk_fail
; LINUX-X64: .cfi_endproc

; LINUX-KERNEL-X64-LABEL: test22a:
; LINUX-KERNEL-X64-NOT: callq __stack_chk_fail
; LINUX-KERNEL-X64: .cfi_endproc

; DARWIN-X64-LABEL: test22a:
; DARWIN-X64-NOT: callq ___stack_chk_fail
; DARWIN-X64: .cfi_endproc

; MSVC-I386-LABEL: test22a:
; MSVC-I386-NOT: calll @__security_check_cookie@4
; MSVC-I386: retl
  %a = alloca %class.A, align 1
  %0 = load i8, ptr %a, align 1
  ret i8 %0
}

; test22b: [2 x i8] in a class
;          ssp attribute
; Requires no protector.
; Function Attrs: ssp
define signext i8 @test22b() #0 {
entry:
; LINUX-I386-LABEL: test22b:
; LINUX-I386-NOT: calll __stack_chk_fail
; LINUX-I386: .cfi_endproc

; LINUX-X64-LABEL: test22b:
; LINUX-X64-NOT: callq __stack_chk_fail
; LINUX-X64: .cfi_endproc

; LINUX-KERNEL-X64-LABEL: test22b:
; LINUX-KERNEL-X64-NOT: callq __stack_chk_fail
; LINUX-KERNEL-X64: .cfi_endproc

; DARWIN-X64-LABEL: test22b:
; DARWIN-X64-NOT: callq ___stack_chk_fail
; DARWIN-X64: .cfi_endproc

; MSVC-I386-LABEL: test22b:
; MSVC-I386-NOT: calll @__security_check_cookie@4
; MSVC-I386: retl
  %a = alloca %class.A, align 1
  %0 = load i8, ptr %a, align 1
  ret i8 %0
}

; test22c: [2 x i8] in a class
;          sspstrong attribute
; Requires protector.
; Function Attrs: sspstrong 
define signext i8 @test22c() #1 {
entry:
; LINUX-I386-LABEL: test22c:
; LINUX-I386: mov{{l|q}} %gs:
; LINUX-I386: calll __stack_chk_fail

; LINUX-X64-LABEL: test22c:
; LINUX-X64: mov{{l|q}} %fs:
; LINUX-X64: callq __stack_chk_fail

; LINUX-KERNEL-X64-LABEL: test22c:
; LINUX-KERNEL-X64: mov{{l|q}} %gs:
; LINUX-KERNEL-X64: callq __stack_chk_fail

; DARWIN-X64-LABEL: test22c:
; DARWIN-X64: mov{{l|q}} ___stack_chk_guard
; DARWIN-X64: callq ___stack_chk_fail

; MSVC-I386-LABEL: test22c:
; MSVC-I386: movl ___security_cookie,
; MSVC-I386: calll @__security_check_cookie@4
  %a = alloca %class.A, align 1
  %0 = load i8, ptr %a, align 1
  ret i8 %0
}

; test22d: [2 x i8] in a class
;          sspreq attribute
; Requires protector.
; Function Attrs: sspreq 
define signext i8 @test22d() #2 {
entry:
; LINUX-I386-LABEL: test22d:
; LINUX-I386: mov{{l|q}} %gs:
; LINUX-I386: calll __stack_chk_fail

; LINUX-X64-LABEL: test22d:
; LINUX-X64: mov{{l|q}} %fs:
; LINUX-X64: callq __stack_chk_fail

; LINUX-KERNEL-X64-LABEL: test22d:
; LINUX-KERNEL-X64: mov{{l|q}} %gs:
; LINUX-KERNEL-X64: callq __stack_chk_fail

; DARWIN-X64-LABEL: test22d:
; DARWIN-X64: mov{{l|q}} ___stack_chk_guard
; DARWIN-X64: callq ___stack_chk_fail

; MSVC-I386-LABEL: test22d:
; MSVC-I386: movl ___security_cookie,
; MSVC-I386: calll @__security_check_cookie@4
  %a = alloca %class.A, align 1
  %0 = load i8, ptr %a, align 1
  ret i8 %0
}

; test23a: [2 x i8] nested in several layers of structs and unions
;          no ssp attribute
; Requires no protector.
define signext i8 @test23a() {
entry:
; LINUX-I386-LABEL: test23a:
; LINUX-I386-NOT: calll __stack_chk_fail
; LINUX-I386: .cfi_endproc

; LINUX-X64-LABEL: test23a:
; LINUX-X64-NOT: callq __stack_chk_fail
; LINUX-X64: .cfi_endproc

; LINUX-KERNEL-X64-LABEL: test23a:
; LINUX-KERNEL-X64-NOT: callq __stack_chk_fail
; LINUX-KERNEL-X64: .cfi_endproc

; DARWIN-X64-LABEL: test23a:
; DARWIN-X64-NOT: callq ___stack_chk_fail
; DARWIN-X64: .cfi_endproc

; MSVC-I386-LABEL: test23a:
; MSVC-I386-NOT: calll @__security_check_cookie@4
; MSVC-I386: retl
  %x = alloca %struct.deep, align 1
  %0 = load i8, ptr %x, align 1
  ret i8 %0
}

; test23b: [2 x i8] nested in several layers of structs and unions
;          ssp attribute
; Requires no protector.
; Function Attrs: ssp
define signext i8 @test23b() #0 {
entry:
; LINUX-I386-LABEL: test23b:
; LINUX-I386-NOT: calll __stack_chk_fail
; LINUX-I386: .cfi_endproc

; LINUX-X64-LABEL: test23b:
; LINUX-X64-NOT: callq __stack_chk_fail
; LINUX-X64: .cfi_endproc

; LINUX-KERNEL-X64-LABEL: test23b:
; LINUX-KERNEL-X64-NOT: callq __stack_chk_fail
; LINUX-KERNEL-X64: .cfi_endproc

; DARWIN-X64-LABEL: test23b:
; DARWIN-X64-NOT: callq ___stack_chk_fail
; DARWIN-X64: .cfi_endproc

; MSVC-I386-LABEL: test23b:
; MSVC-I386-NOT: calll @__security_check_cookie@4
; MSVC-I386: retl
  %x = alloca %struct.deep, align 1
  %0 = load i8, ptr %x, align 1
  ret i8 %0
}

; test23c: [2 x i8] nested in several layers of structs and unions
;          sspstrong attribute
; Requires protector.
; Function Attrs: sspstrong 
define signext i8 @test23c() #1 {
entry:
; LINUX-I386-LABEL: test23c:
; LINUX-I386: mov{{l|q}} %gs:
; LINUX-I386: calll __stack_chk_fail

; LINUX-X64-LABEL: test23c:
; LINUX-X64: mov{{l|q}} %fs:
; LINUX-X64: callq __stack_chk_fail

; LINUX-KERNEL-X64-LABEL: test23c:
; LINUX-KERNEL-X64: mov{{l|q}} %gs:
; LINUX-KERNEL-X64: callq __stack_chk_fail

; DARWIN-X64-LABEL: test23c:
; DARWIN-X64: mov{{l|q}} ___stack_chk_guard
; DARWIN-X64: callq ___stack_chk_fail

; MSVC-I386-LABEL: test23c:
; MSVC-I386: movl ___security_cookie,
; MSVC-I386: calll @__security_check_cookie@4
  %x = alloca %struct.deep, align 1
  %0 = load i8, ptr %x, align 1
  ret i8 %0
}

; test23d: [2 x i8] nested in several layers of structs and unions
;          sspreq attribute
; Requires protector.
; Function Attrs: sspreq 
define signext i8 @test23d() #2 {
entry:
; LINUX-I386-LABEL: test23d:
; LINUX-I386: mov{{l|q}} %gs:
; LINUX-I386: calll __stack_chk_fail

; LINUX-X64-LABEL: test23d:
; LINUX-X64: mov{{l|q}} %fs:
; LINUX-X64: callq __stack_chk_fail

; LINUX-KERNEL-X64-LABEL: test23d:
; LINUX-KERNEL-X64: mov{{l|q}} %gs:
; LINUX-KERNEL-X64: callq __stack_chk_fail

; DARWIN-X64-LABEL: test23d:
; DARWIN-X64: mov{{l|q}} ___stack_chk_guard
; DARWIN-X64: callq ___stack_chk_fail

; MSVC-I386-LABEL: test23d:
; MSVC-I386: movl ___security_cookie,
; MSVC-I386: calll @__security_check_cookie@4
  %x = alloca %struct.deep, align 1
  %0 = load i8, ptr %x, align 1
  ret i8 %0
}

; test24a: Variable sized alloca
;          no ssp attribute
; Requires no protector.
define void @test24a(i32 %n) {
entry:
; LINUX-I386-LABEL: test24a:
; LINUX-I386-NOT: calll __stack_chk_fail
; LINUX-I386: .cfi_endproc

; LINUX-X64-LABEL: test24a:
; LINUX-X64-NOT: callq __stack_chk_fail
; LINUX-X64: .cfi_endproc

; LINUX-KERNEL-X64-LABEL: test24a:
; LINUX-KERNEL-X64-NOT: callq __stack_chk_fail
; LINUX-KERNEL-X64: .cfi_endproc

; DARWIN-X64-LABEL: test24a:
; DARWIN-X64-NOT: callq ___stack_chk_fail
; DARWIN-X64: .cfi_endproc

; MSVC-I386-LABEL: test24a:
; MSVC-I386-NOT: calll @__security_check_cookie@4
; MSVC-I386: retl
  %n.addr = alloca i32, align 4
  %a = alloca ptr, align 8
  store i32 %n, ptr %n.addr, align 4
  %0 = load i32, ptr %n.addr, align 4
  %conv = sext i32 %0 to i64
  %1 = alloca i8, i64 %conv
  store ptr %1, ptr %a, align 8
  ret void
}

; test24b: Variable sized alloca
;          ssp attribute
; Requires protector.
; Function Attrs: ssp
define void @test24b(i32 %n) #0 {
entry:
; LINUX-I386-LABEL: test24b:
; LINUX-I386: mov{{l|q}} %gs:
; LINUX-I386: calll __stack_chk_fail

; LINUX-X64-LABEL: test24b:
; LINUX-X64: mov{{l|q}} %fs:
; LINUX-X64: callq __stack_chk_fail

; LINUX-KERNEL-X64-LABEL: test24b:
; LINUX-KERNEL-X64: mov{{l|q}} %gs:
; LINUX-KERNEL-X64: callq __stack_chk_fail

; DARWIN-X64-LABEL: test24b:
; DARWIN-X64: mov{{l|q}} ___stack_chk_guard
; DARWIN-X64: callq ___stack_chk_fail

; MSVC-I386-LABEL: test24b:
; MSVC-I386: movl ___security_cookie,
; MSVC-I386: calll @__security_check_cookie@4
  %n.addr = alloca i32, align 4
  %a = alloca ptr, align 8
  store i32 %n, ptr %n.addr, align 4
  %0 = load i32, ptr %n.addr, align 4
  %conv = sext i32 %0 to i64
  %1 = alloca i8, i64 %conv
  store ptr %1, ptr %a, align 8
  ret void
}

; test24c: Variable sized alloca
;          sspstrong attribute
; Requires protector.
; Function Attrs: sspstrong 
define void @test24c(i32 %n) #1 {
entry:
; LINUX-I386-LABEL: test24c:
; LINUX-I386: mov{{l|q}} %gs:
; LINUX-I386: calll __stack_chk_fail

; LINUX-X64-LABEL: test24c:
; LINUX-X64: mov{{l|q}} %fs:
; LINUX-X64: callq __stack_chk_fail

; LINUX-KERNEL-X64-LABEL: test24c:
; LINUX-KERNEL-X64: mov{{l|q}} %gs:
; LINUX-KERNEL-X64: callq __stack_chk_fail

; DARWIN-X64-LABEL: test24c:
; DARWIN-X64: mov{{l|q}} ___stack_chk_guard
; DARWIN-X64: callq ___stack_chk_fail

; MSVC-I386-LABEL: test24c:
; MSVC-I386: movl ___security_cookie,
; MSVC-I386: calll @__security_check_cookie@4
  %n.addr = alloca i32, align 4
  %a = alloca ptr, align 8
  store i32 %n, ptr %n.addr, align 4
  %0 = load i32, ptr %n.addr, align 4
  %conv = sext i32 %0 to i64
  %1 = alloca i8, i64 %conv
  store ptr %1, ptr %a, align 8
  ret void
}

; test24d: Variable sized alloca
;          sspreq attribute
; Requires protector.
; Function Attrs: sspreq 
define void @test24d(i32 %n) #2 {
entry:
; LINUX-I386-LABEL: test24d:
; LINUX-I386: mov{{l|q}} %gs:
; LINUX-I386: calll __stack_chk_fail

; LINUX-X64-LABEL: test24d:
; LINUX-X64: mov{{l|q}} %fs:
; LINUX-X64: callq __stack_chk_fail

; LINUX-KERNEL-X64-LABEL: test24d:
; LINUX-KERNEL-X64: mov{{l|q}} %gs:
; LINUX-KERNEL-X64: callq __stack_chk_fail

; DARWIN-X64-LABEL: test24d:
; DARWIN-X64: mov{{l|q}} ___stack_chk_guard
; DARWIN-X64: callq ___stack_chk_fail

; MSVC-I386-LABEL: test24d:
; MSVC-I386: movl ___security_cookie,
; MSVC-I386: calll @__security_check_cookie@4
  %n.addr = alloca i32, align 4
  %a = alloca ptr, align 8
  store i32 %n, ptr %n.addr, align 4
  %0 = load i32, ptr %n.addr, align 4
  %conv = sext i32 %0 to i64
  %1 = alloca i8, i64 %conv
  store ptr %1, ptr %a, align 8
  ret void
}

; test25a: array of [4 x i32]
;          no ssp attribute
; Requires no protector.
define i32 @test25a() {
entry:
; LINUX-I386-LABEL: test25a:
; LINUX-I386-NOT: calll __stack_chk_fail
; LINUX-I386: .cfi_endproc

; LINUX-X64-LABEL: test25a:
; LINUX-X64-NOT: callq __stack_chk_fail
; LINUX-X64: .cfi_endproc

; LINUX-KERNEL-X64-LABEL: test25a:
; LINUX-KERNEL-X64-NOT: callq __stack_chk_fail
; LINUX-KERNEL-X64: .cfi_endproc

; DARWIN-X64-LABEL: test25a:
; DARWIN-X64-NOT: callq ___stack_chk_fail
; DARWIN-X64: .cfi_endproc

; MSVC-I386-LABEL: test25a:
; MSVC-I386-NOT: calll @__security_check_cookie@4
; MSVC-I386: retl
  %a = alloca [4 x i32], align 16
  %0 = load i32, ptr %a, align 4
  ret i32 %0
}

; test25b: array of [4 x i32]
;          ssp attribute
; Requires no protector, except for Darwin which _does_ require a protector.
; Function Attrs: ssp
define i32 @test25b() #0 {
entry:
; LINUX-I386-LABEL: test25b:
; LINUX-I386-NOT: calll __stack_chk_fail
; LINUX-I386: .cfi_endproc

; LINUX-X64-LABEL: test25b:
; LINUX-X64-NOT: callq __stack_chk_fail
; LINUX-X64: .cfi_endproc

; LINUX-KERNEL-X64-LABEL: test25b:
; LINUX-KERNEL-X64-NOT: callq __stack_chk_fail
; LINUX-KERNEL-X64: .cfi_endproc

; DARWIN-X64-LABEL: test25b:
; DARWIN-X64: mov{{l|q}} ___stack_chk_guard
; DARWIN-X64: callq ___stack_chk_fail

; MSVC-I386-LABEL: test25b:
; MSVC-I386-NOT: calll @__security_check_cookie@4
; MSVC-I386: retl

; MINGW-X64-LABEL: test25b:
; MINGW-X64-NOT: callq __stack_chk_fail
; MINGW-X64: .seh_endproc

  %a = alloca [4 x i32], align 16
  %0 = load i32, ptr %a, align 4
  ret i32 %0
}

; test25c: array of [4 x i32]
;          sspstrong attribute
; Requires protector.
; Function Attrs: sspstrong 
define i32 @test25c() #1 {
entry:
; LINUX-I386-LABEL: test25c:
; LINUX-I386: mov{{l|q}} %gs:
; LINUX-I386: calll __stack_chk_fail

; LINUX-X64-LABEL: test25c:
; LINUX-X64: mov{{l|q}} %fs:
; LINUX-X64: callq __stack_chk_fail

; LINUX-KERNEL-X64-LABEL: test25c:
; LINUX-KERNEL-X64: mov{{l|q}} %gs:
; LINUX-KERNEL-X64: callq __stack_chk_fail

; DARWIN-X64-LABEL: test25c:
; DARWIN-X64: mov{{l|q}} ___stack_chk_guard
; DARWIN-X64: callq ___stack_chk_fail

; MSVC-I386-LABEL: test25c:
; MSVC-I386: movl ___security_cookie,
; MSVC-I386: calll @__security_check_cookie@4
  %a = alloca [4 x i32], align 16
  %0 = load i32, ptr %a, align 4
  ret i32 %0
}

; test25d: array of [4 x i32]
;          sspreq attribute
; Requires protector.
; Function Attrs: sspreq 
define i32 @test25d() #2 {
entry:
; LINUX-I386-LABEL: test25d:
; LINUX-I386: mov{{l|q}} %gs:
; LINUX-I386: calll __stack_chk_fail

; LINUX-X64-LABEL: test25d:
; LINUX-X64: mov{{l|q}} %fs:
; LINUX-X64: callq __stack_chk_fail

; LINUX-KERNEL-X64-LABEL: test25d:
; LINUX-KERNEL-X64: mov{{l|q}} %gs:
; LINUX-KERNEL-X64: callq __stack_chk_fail

; DARWIN-X64-LABEL: test25d:
; DARWIN-X64: mov{{l|q}} ___stack_chk_guard
; DARWIN-X64: callq ___stack_chk_fail

; MSVC-I386-LABEL: test25d:
; MSVC-I386: movl ___security_cookie,
; MSVC-I386: calll @__security_check_cookie@4
  %a = alloca [4 x i32], align 16
  %0 = load i32, ptr %a, align 4
  ret i32 %0
}

; test26: Nested structure, no arrays, no address-of expressions.
;         Verify that the resulting gep-of-gep does not incorrectly trigger
;         a stack protector.
;         ssptrong attribute
; Requires no protector.
; Function Attrs: sspstrong 
define void @test26() #1 {
entry:
; LINUX-I386-LABEL: test26:
; LINUX-I386-NOT: calll __stack_chk_fail
; LINUX-I386: .cfi_endproc

; LINUX-X64-LABEL: test26:
; LINUX-X64-NOT: callq __stack_chk_fail
; LINUX-X64: .cfi_endproc

; LINUX-KERNEL-X64-LABEL: test26:
; LINUX-KERNEL-X64-NOT: callq __stack_chk_fail
; LINUX-KERNEL-X64: .cfi_endproc

; DARWIN-X64-LABEL: test26:
; DARWIN-X64-NOT: callq ___stack_chk_fail
; DARWIN-X64: .cfi_endproc

; MSVC-I386-LABEL: test26:
; MSVC-I386-NOT: calll @__security_check_cookie@4
; MSVC-I386: retl
  %c = alloca %struct.nest, align 4
  %b = getelementptr inbounds %struct.nest, ptr %c, i32 0, i32 1
  %0 = load i32, ptr %b, align 4
  %call = call i32 (ptr, ...) @printf(ptr @.str, i32 %0)
  ret void
}

; test27: Address-of a structure taken in a function with a loop where
;         the alloca is an incoming value to a PHI node and a use of that PHI 
;         node is also an incoming value.
;         Verify that the address-of analysis does not get stuck in infinite
;         recursion when chasing the alloca through the PHI nodes.
; Requires protector.
; Function Attrs: sspstrong 
define i32 @test27(i32 %arg) #1 {
bb:
; LINUX-I386-LABEL: test27:
; LINUX-I386: mov{{l|q}} %gs:
; LINUX-I386: calll __stack_chk_fail

; LINUX-X64-LABEL: test27:
; LINUX-X64: mov{{l|q}} %fs:
; LINUX-X64: callq __stack_chk_fail

; LINUX-KERNEL-X64-LABEL: test27:
; LINUX-KERNEL-X64: mov{{l|q}} %gs:
; LINUX-KERNEL-X64: callq __stack_chk_fail

; DARWIN-X64-LABEL: test27:
; DARWIN-X64: mov{{l|q}} ___stack_chk_guard
; DARWIN-X64: callq ___stack_chk_fail

; MSVC-I386-LABEL: test27:
; MSVC-I386: movl ___security_cookie,
; MSVC-I386: calll @__security_check_cookie@4
  %tmp = alloca ptr, align 8
  %tmp1 = call i32 (...) @dummy(ptr %tmp)
  %tmp2 = load ptr, ptr %tmp, align 8
  %tmp3 = ptrtoint ptr %tmp2 to i64
  %tmp4 = trunc i64 %tmp3 to i32
  %tmp5 = icmp sgt i32 %tmp4, 0
  br i1 %tmp5, label %bb6, label %bb21

bb6:                                              ; preds = %bb17, %bb
  %tmp7 = phi ptr [ %tmp19, %bb17 ], [ %tmp2, %bb ]
  %tmp8 = phi i64 [ %tmp20, %bb17 ], [ 1, %bb ]
  %tmp9 = phi i32 [ %tmp14, %bb17 ], [ %tmp1, %bb ]
  %tmp11 = load i8, ptr %tmp7, align 1
  %tmp12 = icmp eq i8 %tmp11, 1
  %tmp13 = add nsw i32 %tmp9, 8
  %tmp14 = select i1 %tmp12, i32 %tmp13, i32 %tmp9
  %tmp15 = trunc i64 %tmp8 to i32
  %tmp16 = icmp eq i32 %tmp15, %tmp4
  br i1 %tmp16, label %bb21, label %bb17

bb17:                                             ; preds = %bb6
  %tmp18 = getelementptr inbounds ptr, ptr %tmp, i64 %tmp8
  %tmp19 = load ptr, ptr %tmp18, align 8
  %tmp20 = add i64 %tmp8, 1
  br label %bb6

bb21:                                             ; preds = %bb6, %bb
  %tmp22 = phi i32 [ %tmp1, %bb ], [ %tmp14, %bb6 ]
  %tmp23 = call i32 (...) @dummy(i32 %tmp22)
  ret i32 undef
}

; test28a: An array of [32 x i8] and a requested ssp-buffer-size of 33.
; Requires no protector.
; Function Attrs: ssp stack-protector-buffer-size=33
define i32 @test28a() #3 {
entry:
; LINUX-I386-LABEL: test28a:
; LINUX-I386-NOT: calll __stack_chk_fail
; LINUX-I386: .cfi_endproc

; LINUX-X64-LABEL: test28a:
; LINUX-X64-NOT: callq __stack_chk_fail
; LINUX-X64: .cfi_endproc

; LINUX-KERNEL-X64-LABEL: test28a:
; LINUX-KERNEL-X64-NOT: callq __stack_chk_fail
; LINUX-KERNEL-X64: .cfi_endproc

; DARWIN-X64-LABEL: test28a:
; DARWIN-X64-NOT: callq ___stack_chk_fail
; DARWIN-X64: .cfi_endproc

; MSVC-I386-LABEL: test28a:
; MSVC-I386-NOT: calll @__security_check_cookie@4
; MSVC-I386: retl
  %test = alloca [32 x i8], align 16
  %call = call i32 (ptr, ...) @printf(ptr @.str, ptr %test)
  ret i32 %call
}

; test28b: An array of [33 x i8] and a requested ssp-buffer-size of 33.
; Requires protector.
; Function Attrs: ssp stack-protector-buffer-size=33
define i32 @test28b() #3 {
entry:
; LINUX-I386-LABEL: test28b:
; LINUX-I386: mov{{l|q}} %gs:
; LINUX-I386: calll __stack_chk_fail

; LINUX-X64-LABEL: test28b:
; LINUX-X64: mov{{l|q}} %fs:
; LINUX-X64: callq __stack_chk_fail

; LINUX-KERNEL-X64-LABEL: test28b:
; LINUX-KERNEL-X64: mov{{l|q}} %gs:
; LINUX-KERNEL-X64: callq __stack_chk_fail

; DARWIN-X64-LABEL: test28b:
; DARWIN-X64: mov{{l|q}} ___stack_chk_guard
; DARWIN-X64: callq ___stack_chk_fail

; MSVC-I386-LABEL: test28b:
; MSVC-I386: movl ___security_cookie,
; MSVC-I386: calll @__security_check_cookie@4
  %test = alloca [33 x i8], align 16
  %call = call i32 (ptr, ...) @printf(ptr @.str, ptr %test)
  ret i32 %call
}

; test29a: An array of [4 x i8] and a requested ssp-buffer-size of 5.
; Requires no protector.
; Function Attrs: ssp stack-protector-buffer-size=5
define i32 @test29a() #4 {
entry:
; LINUX-I386-LABEL: test29a:
; LINUX-I386-NOT: calll __stack_chk_fail
; LINUX-I386: .cfi_endproc

; LINUX-X64-LABEL: test29a:
; LINUX-X64-NOT: callq __stack_chk_fail
; LINUX-X64: .cfi_endproc

; LINUX-KERNEL-X64-LABEL: test29a:
; LINUX-KERNEL-X64-NOT: callq __stack_chk_fail
; LINUX-KERNEL-X64: .cfi_endproc

; DARWIN-X64-LABEL: test29a:
; DARWIN-X64-NOT: callq ___stack_chk_fail
; DARWIN-X64: .cfi_endproc

; MSVC-I386-LABEL: test29a:
; MSVC-I386-NOT: calll @__security_check_cookie@4
; MSVC-I386: retl
  %test = alloca [4 x i8], align 1
  %call = call i32 (ptr, ...) @printf(ptr @.str, ptr %test)
  ret i32 %call
}

; test29b: An array of [5 x i8] and a requested ssp-buffer-size of 5.
; Requires protector.
; Function Attrs: ssp stack-protector-buffer-size=5
define i32 @test29b() #4 {
entry:
; LINUX-I386-LABEL: test29b:
; LINUX-I386: mov{{l|q}} %gs:
; LINUX-I386: calll __stack_chk_fail

; LINUX-X64-LABEL: test29b:
; LINUX-X64: mov{{l|q}} %fs:
; LINUX-X64: callq __stack_chk_fail

; LINUX-KERNEL-X64-LABEL: test29b:
; LINUX-KERNEL-X64: mov{{l|q}} %gs:
; LINUX-KERNEL-X64: callq __stack_chk_fail

; DARWIN-X64-LABEL: test29b:
; DARWIN-X64: mov{{l|q}} ___stack_chk_guard
; DARWIN-X64: callq ___stack_chk_fail

; MSVC-I386-LABEL: test29b:
; MSVC-I386: movl ___security_cookie,
; MSVC-I386: calll @__security_check_cookie@4
  %test = alloca [5 x i8], align 1
  %call = call i32 (ptr, ...) @printf(ptr @.str, ptr %test)
  ret i32 %call
}

; test30a: An structure containing an i32 and an array of [5 x i8].
;          Requested ssp-buffer-size of 6.
; Requires no protector.
; Function Attrs: ssp stack-protector-buffer-size=6
define i32 @test30a() #5 {
entry:
; LINUX-I386-LABEL: test30a:
; LINUX-I386-NOT: calll __stack_chk_fail
; LINUX-I386: .cfi_endproc

; LINUX-X64-LABEL: test30a:
; LINUX-X64-NOT: callq __stack_chk_fail
; LINUX-X64: .cfi_endproc

; LINUX-KERNEL-X64-LABEL: test30a:
; LINUX-KERNEL-X64-NOT: callq __stack_chk_fail
; LINUX-KERNEL-X64: .cfi_endproc

; DARWIN-X64-LABEL: test30a:
; DARWIN-X64-NOT: callq ___stack_chk_fail
; DARWIN-X64: .cfi_endproc

; MSVC-I386-LABEL: test30a:
; MSVC-I386-NOT: calll @__security_check_cookie@4
; MSVC-I386: retl
  %test = alloca %struct.small_char, align 4
  %test.coerce = alloca { i64, i8 }
  call void @llvm.memcpy.p0.p0.i64(ptr %test.coerce, ptr %test, i64 12, i1 false)
  %0 = getelementptr { i64, i8 }, ptr %test.coerce, i32 0, i32 0
  %1 = load i64, ptr %0, align 1
  %2 = getelementptr { i64, i8 }, ptr %test.coerce, i32 0, i32 1
  %3 = load i8, ptr %2, align 1
  %call = call i32 (ptr, ...) @printf(ptr @.str, i64 %1, i8 %3)
  ret i32 %call
}

; test30b: An structure containing an i32 and an array of [5 x i8].
;          Requested ssp-buffer-size of 5.
; Requires protector.
; Function Attrs: ssp stack-protector-buffer-size=5
define i32 @test30b() #4 {
entry:
; LINUX-I386-LABEL: test30b:
; LINUX-I386: mov{{l|q}} %gs:
; LINUX-I386: calll __stack_chk_fail

; LINUX-X64-LABEL: test30b:
; LINUX-X64: mov{{l|q}} %fs:
; LINUX-X64: callq __stack_chk_fail

; LINUX-KERNEL-X64-LABEL: test30b:
; LINUX-KERNEL-X64: mov{{l|q}} %gs:
; LINUX-KERNEL-X64: callq __stack_chk_fail

; DARWIN-X64-LABEL: test30b:
; DARWIN-X64: mov{{l|q}} ___stack_chk_guard
; DARWIN-X64: callq ___stack_chk_fail

; MSVC-I386-LABEL: test30b:
; MSVC-I386: movl ___security_cookie,
; MSVC-I386: calll @__security_check_cookie@4
  %test = alloca %struct.small_char, align 4
  %test.coerce = alloca { i64, i8 }
  call void @llvm.memcpy.p0.p0.i64(ptr %test.coerce, ptr %test, i64 12, i1 false)
  %0 = getelementptr { i64, i8 }, ptr %test.coerce, i32 0, i32 0
  %1 = load i64, ptr %0, align 1
  %2 = getelementptr { i64, i8 }, ptr %test.coerce, i32 0, i32 1
  %3 = load i8, ptr %2, align 1
  %call = call i32 (ptr, ...) @printf(ptr @.str, i64 %1, i8 %3)
  ret i32 %call
}

; test31a: An alloca of size 5.
;          Requested ssp-buffer-size of 6.
; Requires no protector.
; Function Attrs: ssp stack-protector-buffer-size=6
define i32 @test31a() #5 {
entry:
; LINUX-I386-LABEL: test31a:
; LINUX-I386-NOT: calll __stack_chk_fail
; LINUX-I386: .cfi_endproc

; LINUX-X64-LABEL: test31a:
; LINUX-X64-NOT: callq __stack_chk_fail
; LINUX-X64: .cfi_endproc

; LINUX-KERNEL-X64-LABEL: test31a:
; LINUX-KERNEL-X64-NOT: callq __stack_chk_fail
; LINUX-KERNEL-X64: .cfi_endproc

; DARWIN-X64-LABEL: test31a:
; DARWIN-X64-NOT: callq ___stack_chk_fail
; DARWIN-X64: .cfi_endproc

; MSVC-I386-LABEL: test31a:
; MSVC-I386-NOT: calll @__security_check_cookie@4
; MSVC-I386: retl
  %test = alloca ptr, align 8
  %0 = alloca i8, i64 4
  store ptr %0, ptr %test, align 8
  %1 = load ptr, ptr %test, align 8
  %call = call i32 (ptr, ...) @printf(ptr @.str, ptr %1)
  ret i32 %call
}

; test31b: An alloca of size 5.
;          Requested ssp-buffer-size of 5.
; Requires protector.
define i32 @test31b() #4 {
entry:
; LINUX-I386-LABEL: test31b:
; LINUX-I386: mov{{l|q}} %gs:
; LINUX-I386: calll __stack_chk_fail

; LINUX-X64-LABEL: test31b:
; LINUX-X64: mov{{l|q}} %fs:
; LINUX-X64: callq __stack_chk_fail

; LINUX-KERNEL-X64-LABEL: test31b:
; LINUX-KERNEL-X64: mov{{l|q}} %gs:
; LINUX-KERNEL-X64: callq __stack_chk_fail

; DARWIN-X64-LABEL: test31b:
; DARWIN-X64: mov{{l|q}} ___stack_chk_guard
; DARWIN-X64: callq ___stack_chk_fail

; MSVC-I386-LABEL: test31b:
; MSVC-I386: movl ___security_cookie,
; MSVC-I386: calll @__security_check_cookie@4
  %test = alloca ptr, align 8
  %0 = alloca i8, i64 5
  store ptr %0, ptr %test, align 8
  %1 = load ptr, ptr %test, align 8
  %call = call i32 (ptr, ...) @printf(ptr @.str, ptr %1)
  ret i32 %call
}

define void @__stack_chk_fail() #1 !dbg !6 {
entry:
  ret void
}

define void @test32() #1 !dbg !7 {
entry:
; LINUX-I386-LABEL: test32:
; LINUX-I386:       .loc 1 4 2 prologue_end
; LINUX-I386:       .loc 1 0 0
; LINUX-I386-NEXT:  calll __stack_chk_fail

; LINUX-X64-LABEL: test32:
; LINUX-X64:       .loc 1 4 2 prologue_end
; LINUX-X64:       .loc 1 0 0
; LINUX-X64-NEXT:  callq __stack_chk_fail

; LINUX-KERNEL-X64-LABEL: test32:
; LINUX-KERNEL-X64:       .loc 1 4 2 prologue_end
; LINUX-KERNEL-X64:       .loc 1 0 0
; LINUX-KERNEL-X64-NEXT:  callq __stack_chk_fail

; OPENBSD-AMD64-LABEL: test32:
; OPENBSD-AMD64:       .loc 1 4 2 prologue_end
; OPENBSD-AMD64:       .loc 1 0 0
; OPENBSD-AMD64-NEXT:  movl
; OPENBSD-AMD64-NEXT:  callq __stack_smash_handler
  %0 = alloca [5 x i8], align 1
  ret void, !dbg !9
}

define i32 @IgnoreIntrinsicTest() #1 {
; IGNORE_INTRIN: IgnoreIntrinsicTest:
  %1 = alloca i32, align 4
  call void @llvm.lifetime.start.p0(i64 4, ptr nonnull %1)
  store volatile i32 1, ptr %1, align 4
  %2 = load volatile i32, ptr %1, align 4
  %3 = mul nsw i32 %2, 42
  call void @llvm.lifetime.end.p0(i64 4, ptr nonnull %1)
  ret i32 %3
; IGNORE_INTRIN-NOT: callq __stack_chk_fail
; IGNORE_INTRIN:     .cfi_endproc
}

declare double @testi_aux()
declare ptr @strcpy(ptr, ptr)
declare i32 @printf(ptr, ...)
declare void @funcall(ptr)
declare void @funcall2(ptr)
declare void @funfloat(ptr)
declare void @funfloat2(ptr)
declare void @_Z3exceptPi(ptr)
declare i32 @__gxx_personality_v0(...)
declare ptr @getp()
declare i32 @dummy(...)
declare void @llvm.memcpy.p0.p0.i64(ptr nocapture, ptr nocapture readonly, i64, i1)
declare void @llvm.lifetime.start.p0(i64, ptr nocapture)
declare void @llvm.lifetime.end.p0(i64, ptr nocapture)

attributes #0 = { ssp }
attributes #1 = { sspstrong }
attributes #2 = { sspreq }
attributes #3 = { ssp "stack-protector-buffer-size"="33" }
attributes #4 = { ssp "stack-protector-buffer-size"="5" }
attributes #5 = { ssp "stack-protector-buffer-size"="6" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!llvm.ident = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, emissionKind: FullDebug)
!1 = !DIFile(filename: "test.c", directory: "/tmp")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{!"clang version x.y.z"}
!6 = distinct !DISubprogram(name: "__stack_chk_fail", scope: !1, type: !8, unit: !0)
!7 = distinct !DISubprogram(name: "test32", scope: !1, type: !8, unit: !0)
!8 = !DISubroutineType(types: !2)
!9 = !DILocation(line: 4, column: 2, scope: !7)
