; RUN: llc -mtriple=i686-windows < %s | FileCheck %s

declare void @addrof_i1(i1*)
declare void @addrof_i32(i32*)
declare void @addrof_i64(i64*)
declare void @addrof_i128(i128*)
declare void @addrof_i32_x3(i32*, i32*, i32*)

define void @simple(i32 %x) {
entry:
  %x.addr = alloca i32
  store i32 %x, i32* %x.addr
  call void @addrof_i32(i32* %x.addr)
  ret void
}

; CHECK-LABEL: _simple:
; CHECK: leal 4(%esp), %[[reg:[^ ]*]]
; CHECK: pushl %[[reg]]
; CHECK: calll _addrof_i32
; CHECK: retl


; We need to load %x before calling addrof_i32 now because it could mutate %x in
; place.

define i32 @use_arg(i32 %x) {
entry:
  %x.addr = alloca i32
  store i32 %x, i32* %x.addr
  call void @addrof_i32(i32* %x.addr)
  ret i32 %x
}

; CHECK-LABEL: _use_arg:
; CHECK: pushl %[[csr:[^ ]*]]
; CHECK-DAG: movl 8(%esp), %[[csr]]
; CHECK-DAG: leal 8(%esp), %[[reg:[^ ]*]]
; CHECK: pushl %[[reg]]
; CHECK: calll _addrof_i32
; CHECK: movl %[[csr]], %eax
; CHECK: popl %[[csr]]
; CHECK: retl

; We won't copy elide for types needing legalization such as i64 or i1.

define i64 @split_i64(i64 %x) {
entry:
  %x.addr = alloca i64, align 4
  store i64 %x, i64* %x.addr, align 4
  call void @addrof_i64(i64* %x.addr)
  ret i64 %x
}

; CHECK-LABEL: _split_i64:
; CHECK: pushl %ebp
; CHECK: movl %esp, %ebp
; CHECK: pushl %[[csr2:[^ ]*]]
; CHECK: pushl %[[csr1:[^ ]*]]
; CHECK: andl $-8, %esp
; CHECK-DAG: movl 8(%ebp), %[[csr1]]
; CHECK-DAG: movl 12(%ebp), %[[csr2]]
; CHECK: movl %edi, 4(%esp)
; CHECK: movl %esi, (%esp)
; CEHCK: movl %esp, %eax
; CHECK: pushl %eax
; CHECK: calll _addrof_i64
; CHECK-DAG: movl %[[csr1]], %eax
; CHECK-DAG: movl %[[csr2]], %edx
; CHECK: leal -8(%ebp), %esp
; CHECK: popl %[[csr1]]
; CHECK: popl %[[csr2]]
; CHECK: popl %ebp
; CHECK: retl

define i1 @i1_arg(i1 %x) {
  %x.addr = alloca i1
  store i1 %x, i1* %x.addr
  call void @addrof_i1(i1* %x.addr)
  ret i1 %x
}

; CHECK-LABEL: _i1_arg:
; CHECK: pushl   %ebx
; CHECK: movb 8(%esp), %bl
; CHECK: leal 8(%esp), %eax
; CHECK: pushl %eax
; CHECK: calll _addrof_i1
; CHECK: addl $4, %esp
; CHECK: movl %ebx, %eax
; CHECK: popl %ebx
; CHECK: retl

; We can't copy elide when an i64 is split between registers and memory in a
; fastcc function.

define fastcc i64 @fastcc_split_i64(i64* %p, i64 %x) {
entry:
  %x.addr = alloca i64, align 4
  store i64 %x, i64* %x.addr, align 4
  call void @addrof_i64(i64* %x.addr)
  ret i64 %x
}

; CHECK-LABEL: _fastcc_split_i64:
; CHECK: pushl %ebp
; CHECK: movl %esp, %ebp
; CHECK-DAG: movl %edx, %[[r1:[^ ]*]]
; CHECK-DAG: movl 8(%ebp), %[[r2:[^ ]*]]
; CHECK-DAG: movl %[[r2]], 4(%esp)
; CHECK-DAG: movl %[[r1]], (%esp)
; CHECK: movl %esp, %[[reg:[^ ]*]]
; CHECK: pushl %[[reg]]
; CHECK: calll _addrof_i64
; CHECK: popl %ebp
; CHECK: retl


; We can't copy elide when it would reduce the user requested alignment.

define void @high_alignment(i32 %x) {
entry:
  %x.p = alloca i32, align 128
  store i32 %x, i32* %x.p
  call void @addrof_i32(i32* %x.p)
  ret void
}

; CHECK-LABEL: _high_alignment:
; CHECK: andl $-128, %esp
; CHECK: movl 8(%ebp), %[[reg:[^ ]*]]
; CHECK: movl %[[reg]], (%esp)
; CHECK: movl %esp, %[[reg:[^ ]*]]
; CHECK: pushl %[[reg]]
; CHECK: calll _addrof_i32
; CHECK: retl


; We can't copy elide when it would reduce the ABI required alignment.
; FIXME: We should lower the ABI alignment of i64 on Windows, since MSVC
; doesn't guarantee it.

define void @abi_alignment(i64 %x) {
entry:
  %x.p = alloca i64
  store i64 %x, i64* %x.p
  call void @addrof_i64(i64* %x.p)
  ret void
}

; CHECK-LABEL: _abi_alignment:
; CHECK: andl $-8, %esp
; CHECK: movl 8(%ebp), %[[reg:[^ ]*]]
; CHECK: movl %[[reg]], (%esp)
; CHECK: movl %esp, %[[reg:[^ ]*]]
; CHECK: pushl %[[reg]]
; CHECK: calll _addrof_i64
; CHECK: retl


; The code we generate for this is unimportant. This is mostly a crash test.

define void @split_i128(i128* %sret, i128 %x) {
entry:
  %x.addr = alloca i128
  store i128 %x, i128* %x.addr
  call void @addrof_i128(i128* %x.addr)
  store i128 %x, i128* %sret
  ret void
}

; CHECK-LABEL: _split_i128:
; CHECK: pushl %ebp
; CHECK: calll _addrof_i128
; CHECK: retl


; Check that we load all of x, y, and z before the call.

define i32 @three_args(i32 %x, i32 %y, i32 %z) {
entry:
  %z.addr = alloca i32, align 4
  %y.addr = alloca i32, align 4
  %x.addr = alloca i32, align 4
  store i32 %z, i32* %z.addr, align 4
  store i32 %y, i32* %y.addr, align 4
  store i32 %x, i32* %x.addr, align 4
  call void @addrof_i32_x3(i32* %x.addr, i32* %y.addr, i32* %z.addr)
  %s1 = add i32 %x, %y
  %sum = add i32 %s1, %z
  ret i32 %sum
}

; CHECK-LABEL: _three_args:
; CHECK: pushl %[[csr:[^ ]*]]
; CHECK-DAG: movl {{[0-9]+}}(%esp), %[[csr]]
; CHECK-DAG: addl {{[0-9]+}}(%esp), %[[csr]]
; CHECK-DAG: addl {{[0-9]+}}(%esp), %[[csr]]
; CHECK-DAG: leal 8(%esp), %[[x:[^ ]*]]
; CHECK-DAG: leal 12(%esp), %[[y:[^ ]*]]
; CHECK-DAG: leal 16(%esp), %[[z:[^ ]*]]
; CHECK: pushl %[[z]]
; CHECK: pushl %[[y]]
; CHECK: pushl %[[x]]
; CHECK: calll _addrof_i32_x3
; CHECK: movl %[[csr]], %eax
; CHECK: popl %[[csr]]
; CHECK: retl


define void @two_args_same_alloca(i32 %x, i32 %y) {
entry:
  %x.addr = alloca i32
  store i32 %x, i32* %x.addr
  store i32 %y, i32* %x.addr
  call void @addrof_i32(i32* %x.addr)
  ret void
}

; CHECK-LABEL: _two_args_same_alloca:
; CHECK: movl 8(%esp), {{.*}}
; CHECK: movl {{.*}}, 4(%esp)
; CHECK: leal 4(%esp), %[[reg:[^ ]*]]
; CHECK: pushl %[[reg]]
; CHECK: calll _addrof_i32
; CHECK: retl


define void @avoid_byval(i32* byval %x) {
entry:
  %x.p.p = alloca i32*
  store i32* %x, i32** %x.p.p
  call void @addrof_i32(i32* %x)
  ret void
}

; CHECK-LABEL: _avoid_byval:
; CHECK: leal {{[0-9]+}}(%esp), %[[reg:[^ ]*]]
; CHECK: pushl %[[reg]]
; CHECK: calll _addrof_i32
; CHECK: retl


define void @avoid_inalloca(i32* inalloca %x) {
entry:
  %x.p.p = alloca i32*
  store i32* %x, i32** %x.p.p
  call void @addrof_i32(i32* %x)
  ret void
}

; CHECK-LABEL: _avoid_inalloca:
; CHECK: leal {{[0-9]+}}(%esp), %[[reg:[^ ]*]]
; CHECK: pushl %[[reg]]
; CHECK: calll _addrof_i32
; CHECK: retl


; Don't elide the copy when the alloca is escaped with a store.

define void @escape_with_store(i32 %x) {
  %x1 = alloca i32
  %x2 = alloca i32*
  store i32* %x1, i32** %x2
  %x3 = load i32*, i32** %x2
  store i32 0, i32* %x3
  store i32 %x, i32* %x1
  call void @addrof_i32(i32* %x1)
  ret void
}

; CHECK-LABEL: _escape_with_store:
; CHECK-DAG: movl {{.*}}(%esp), %[[reg:[^ ]*]]
; CHECK-DAG: movl $0, [[offs:[0-9]*]](%esp)
; CHECK: movl %[[reg]], [[offs]](%esp)
; CHECK: calll _addrof_i32


; This test case exposed issues with the use of TokenFactor.

define void @sret_and_elide(i32* sret %sret, i32 %v) {
  %v.p = alloca i32
  store i32 %v, i32* %v.p
  call void @addrof_i32(i32* %v.p)
  store i32 %v, i32* %sret
  ret void
}

; CHECK-LABEL: _sret_and_elide:
; CHECK: pushl
; CHECK: pushl
; CHECK: movl 12(%esp), %[[sret:[^ ]*]]
; CHECK: movl 16(%esp), %[[v:[^ ]*]]
; CHECK: leal 16(%esp), %[[reg:[^ ]*]]
; CHECK: pushl %[[reg]]
; CHECK: calll _addrof_i32
; CHECK: movl %[[v]], (%[[sret]])
; CHECK: movl %[[sret]], %eax
; CHECK: popl
; CHECK: popl
; CHECK: retl
