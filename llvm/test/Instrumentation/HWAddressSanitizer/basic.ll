; Test basic address sanitizer instrumentation.
;
; RUN: opt < %s -hwasan -hwasan-recover=0 -hwasan-with-ifunc=1 -hwasan-with-tls=0 -S | FileCheck %s --check-prefixes=CHECK,ABORT,ABORT-DYNAMIC-SHADOW
; RUN: opt < %s -hwasan -hwasan-recover=1 -hwasan-with-ifunc=1 -hwasan-with-tls=0 -S | FileCheck %s --check-prefixes=CHECK,RECOVER,RECOVER-DYNAMIC-SHADOW
; RUN: opt < %s -hwasan -hwasan-recover=0 -hwasan-mapping-offset=0 -S | FileCheck %s --check-prefixes=CHECK,ABORT,ABORT-ZERO-BASED-SHADOW
; RUN: opt < %s -hwasan -hwasan-recover=1 -hwasan-mapping-offset=0 -S | FileCheck %s --check-prefixes=CHECK,RECOVER,RECOVER-ZERO-BASED-SHADOW

; Ensure than hwasan runs with the new PM pass
; RUN: opt < %s -passes=hwasan -hwasan-recover=0 -hwasan-with-ifunc=1 -hwasan-with-tls=0 -S | FileCheck %s --check-prefixes=CHECK,ABORT,ABORT-DYNAMIC-SHADOW
; RUN: opt < %s -passes=hwasan -hwasan-recover=1 -hwasan-with-ifunc=1 -hwasan-with-tls=0 -S | FileCheck %s --check-prefixes=CHECK,RECOVER,RECOVER-DYNAMIC-SHADOW
; RUN: opt < %s -passes=hwasan -hwasan-recover=0 -hwasan-mapping-offset=0 -S | FileCheck %s --check-prefixes=CHECK,ABORT,ABORT-ZERO-BASED-SHADOW
; RUN: opt < %s -passes=hwasan -hwasan-recover=1 -hwasan-mapping-offset=0 -S | FileCheck %s --check-prefixes=CHECK,RECOVER,RECOVER-ZERO-BASED-SHADOW

; CHECK: @llvm.global_ctors = appending global [1 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 0, void ()* @hwasan.module_ctor, i8* bitcast (void ()* @hwasan.module_ctor to i8*) }]

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64--linux-android10000"

define i8 @test_load8(i8* %a) sanitize_hwaddress {
; CHECK-LABEL: @test_load8(
; RECOVER: %[[A:[^ ]*]] = ptrtoint i8* %a to i64
; RECOVER: %[[B:[^ ]*]] = lshr i64 %[[A]], 56
; RECOVER: %[[PTRTAG:[^ ]*]] = trunc i64 %[[B]] to i8
; RECOVER: %[[C:[^ ]*]] = and i64 %[[A]], 72057594037927935
; RECOVER: %[[D:[^ ]*]] = lshr i64 %[[C]], 4
; RECOVER-DYNAMIC-SHADOW: %[[E:[^ ]*]] = getelementptr i8, i8* %.hwasan.shadow, i64 %4
; RECOVER-ZERO-BASED-SHADOW: %[[E:[^ ]*]] = inttoptr i64 %[[D]] to i8*
; RECOVER: %[[MEMTAG:[^ ]*]] = load i8, i8* %[[E]]
; RECOVER: %[[F:[^ ]*]] = icmp ne i8 %[[PTRTAG]], %[[MEMTAG]]
; RECOVER: br i1 %[[F]], label %[[MISMATCH:[0-9]*]], label %[[CONT:[0-9]*]], !prof {{.*}}

; RECOVER: [[MISMATCH]]:
; RECOVER: %[[NOTSHORT:[^ ]*]] = icmp ugt i8 %[[MEMTAG]], 15
; RECOVER: br i1 %[[NOTSHORT]], label %[[FAIL:[0-9]*]], label %[[SHORT:[0-9]*]], !prof {{.*}}

; RECOVER: [[FAIL]]:
; RECOVER: call void asm sideeffect "brk #2336", "{x0}"(i64 %[[A]])
; RECOVER: br label

; RECOVER: [[SHORT]]:
; RECOVER: %[[LOWBITS:[^ ]*]] = and i64 %[[A]], 15
; RECOVER: %[[LOWBITS_I8:[^ ]*]] = trunc i64 %[[LOWBITS]] to i8
; RECOVER: %[[LAST:[^ ]*]] = add i8 %[[LOWBITS_I8]], 0
; RECOVER: %[[OOB:[^ ]*]] = icmp uge i8 %[[LAST]], %[[MEMTAG]]
; RECOVER: br i1 %[[OOB]], label %[[FAIL]], label %[[INBOUNDS:[0-9]*]], !prof {{.*}}

; RECOVER: [[INBOUNDS]]:
; RECOVER: %[[EOG_ADDR:[^ ]*]] = or i64 %[[C]], 15
; RECOVER: %[[EOG_PTR:[^ ]*]] = inttoptr i64 %[[EOG_ADDR]] to i8*
; RECOVER: %[[EOGTAG:[^ ]*]] = load i8, i8* %[[EOG_PTR]]
; RECOVER: %[[EOG_MISMATCH:[^ ]*]] = icmp ne i8 %[[PTRTAG]], %[[EOGTAG]]
; RECOVER: br i1 %[[EOG_MISMATCH]], label %[[FAIL]], label %[[CONT1:[0-9]*]], !prof {{.*}}

; RECOVER: [[CONT1]]:
; RECOVER: br label %[[CONT]]

; RECOVER: [[CONT]]:

; ABORT: call void @llvm.hwasan.check.memaccess.shortgranules(i8* %.hwasan.shadow, i8* %a, i32 0)

; CHECK: %[[G:[^ ]*]] = load i8, i8* %a, align 4
; CHECK: ret i8 %[[G]]

entry:
  %b = load i8, i8* %a, align 4
  ret i8 %b
}

define i16 @test_load16(i16* %a) sanitize_hwaddress {
; CHECK-LABEL: @test_load16(
; RECOVER: %[[A:[^ ]*]] = ptrtoint i16* %a to i64
; RECOVER: %[[B:[^ ]*]] = lshr i64 %[[A]], 56
; RECOVER: %[[PTRTAG:[^ ]*]] = trunc i64 %[[B]] to i8
; RECOVER: %[[C:[^ ]*]] = and i64 %[[A]], 72057594037927935
; RECOVER: %[[D:[^ ]*]] = lshr i64 %[[C]], 4
; RECOVER-DYNAMIC-SHADOW: %[[E:[^ ]*]] = getelementptr i8, i8* %.hwasan.shadow, i64 %4
; RECOVER-ZERO-BASED-SHADOW: %[[E:[^ ]*]] = inttoptr i64 %[[D]] to i8*
; RECOVER: %[[MEMTAG:[^ ]*]] = load i8, i8* %[[E]]
; RECOVER: %[[F:[^ ]*]] = icmp ne i8 %[[PTRTAG]], %[[MEMTAG]]
; RECOVER: br i1 %[[F]], label %[[MISMATCH:[0-9]*]], label %[[CONT:[0-9]*]], !prof {{.*}}

; RECOVER: [[MISMATCH]]:
; RECOVER: %[[NOTSHORT:[^ ]*]] = icmp ugt i8 %[[MEMTAG]], 15
; RECOVER: br i1 %[[NOTSHORT]], label %[[FAIL:[0-9]*]], label %[[SHORT:[0-9]*]], !prof {{.*}}

; RECOVER: [[FAIL]]:
; RECOVER: call void asm sideeffect "brk #2337", "{x0}"(i64 %[[A]])
; RECOVER: br label

; RECOVER: [[SHORT]]:
; RECOVER: %[[LOWBITS:[^ ]*]] = and i64 %[[A]], 15
; RECOVER: %[[LOWBITS_I8:[^ ]*]] = trunc i64 %[[LOWBITS]] to i8
; RECOVER: %[[LAST:[^ ]*]] = add i8 %[[LOWBITS_I8]], 1
; RECOVER: %[[OOB:[^ ]*]] = icmp uge i8 %[[LAST]], %[[MEMTAG]]
; RECOVER: br i1 %[[OOB]], label %[[FAIL]], label %[[INBOUNDS:[0-9]*]], !prof {{.*}}

; RECOVER: [[INBOUNDS]]:
; RECOVER: %[[EOG_ADDR:[^ ]*]] = or i64 %[[C]], 15
; RECOVER: %[[EOG_PTR:[^ ]*]] = inttoptr i64 %[[EOG_ADDR]] to i8*
; RECOVER: %[[EOGTAG:[^ ]*]] = load i8, i8* %[[EOG_PTR]]
; RECOVER: %[[EOG_MISMATCH:[^ ]*]] = icmp ne i8 %[[PTRTAG]], %[[EOGTAG]]
; RECOVER: br i1 %[[EOG_MISMATCH]], label %[[FAIL]], label %[[CONT1:[0-9]*]], !prof {{.*}}

; RECOVER: [[CONT1]]:
; RECOVER: br label %[[CONT]]

; RECOVER: [[CONT]]:

; ABORT: %[[A:[^ ]*]] = bitcast i16* %a to i8*
; ABORT: call void @llvm.hwasan.check.memaccess.shortgranules(i8* %.hwasan.shadow, i8* %[[A]], i32 1)

; CHECK: %[[G:[^ ]*]] = load i16, i16* %a, align 4
; CHECK: ret i16 %[[G]]

entry:
  %b = load i16, i16* %a, align 4
  ret i16 %b
}

define i32 @test_load32(i32* %a) sanitize_hwaddress {
; CHECK-LABEL: @test_load32(
; RECOVER: %[[A:[^ ]*]] = ptrtoint i32* %a to i64
; RECOVER: %[[B:[^ ]*]] = lshr i64 %[[A]], 56
; RECOVER: %[[PTRTAG:[^ ]*]] = trunc i64 %[[B]] to i8
; RECOVER: %[[C:[^ ]*]] = and i64 %[[A]], 72057594037927935
; RECOVER: %[[D:[^ ]*]] = lshr i64 %[[C]], 4
; RECOVER-DYNAMIC-SHADOW: %[[E:[^ ]*]] = getelementptr i8, i8* %.hwasan.shadow, i64 %4
; RECOVER-ZERO-BASED-SHADOW: %[[E:[^ ]*]] = inttoptr i64 %[[D]] to i8*
; RECOVER: %[[MEMTAG:[^ ]*]] = load i8, i8* %[[E]]
; RECOVER: %[[F:[^ ]*]] = icmp ne i8 %[[PTRTAG]], %[[MEMTAG]]
; RECOVER: br i1 %[[F]], label {{.*}}, label {{.*}}, !prof {{.*}}

; RECOVER: call void asm sideeffect "brk #2338", "{x0}"(i64 %[[A]])
; RECOVER: br label

; ABORT: %[[A:[^ ]*]] = bitcast i32* %a to i8*
; ABORT: call void @llvm.hwasan.check.memaccess.shortgranules(i8* %.hwasan.shadow, i8* %[[A]], i32 2)

; CHECK: %[[G:[^ ]*]] = load i32, i32* %a, align 4
; CHECK: ret i32 %[[G]]

entry:
  %b = load i32, i32* %a, align 4
  ret i32 %b
}

define i64 @test_load64(i64* %a) sanitize_hwaddress {
; CHECK-LABEL: @test_load64(
; RECOVER: %[[A:[^ ]*]] = ptrtoint i64* %a to i64
; RECOVER: %[[B:[^ ]*]] = lshr i64 %[[A]], 56
; RECOVER: %[[PTRTAG:[^ ]*]] = trunc i64 %[[B]] to i8
; RECOVER: %[[C:[^ ]*]] = and i64 %[[A]], 72057594037927935
; RECOVER: %[[D:[^ ]*]] = lshr i64 %[[C]], 4
; RECOVER-DYNAMIC-SHADOW: %[[E:[^ ]*]] = getelementptr i8, i8* %.hwasan.shadow, i64 %4
; RECOVER-ZERO-BASED-SHADOW: %[[E:[^ ]*]] = inttoptr i64 %[[D]] to i8*
; RECOVER: %[[MEMTAG:[^ ]*]] = load i8, i8* %[[E]]
; RECOVER: %[[F:[^ ]*]] = icmp ne i8 %[[PTRTAG]], %[[MEMTAG]]
; RECOVER: br i1 %[[F]], label {{.*}}, label {{.*}}, !prof {{.*}}

; RECOVER: call void asm sideeffect "brk #2339", "{x0}"(i64 %[[A]])
; RECOVER: br label

; ABORT: %[[A:[^ ]*]] = bitcast i64* %a to i8*
; ABORT: call void @llvm.hwasan.check.memaccess.shortgranules(i8* %.hwasan.shadow, i8* %[[A]], i32 3)

; CHECK: %[[G:[^ ]*]] = load i64, i64* %a, align 8
; CHECK: ret i64 %[[G]]

entry:
  %b = load i64, i64* %a, align 8
  ret i64 %b
}

define i128 @test_load128(i128* %a) sanitize_hwaddress {
; CHECK-LABEL: @test_load128(
; RECOVER: %[[A:[^ ]*]] = ptrtoint i128* %a to i64
; RECOVER: %[[B:[^ ]*]] = lshr i64 %[[A]], 56
; RECOVER: %[[PTRTAG:[^ ]*]] = trunc i64 %[[B]] to i8
; RECOVER: %[[C:[^ ]*]] = and i64 %[[A]], 72057594037927935
; RECOVER: %[[D:[^ ]*]] = lshr i64 %[[C]], 4
; RECOVER-DYNAMIC-SHADOW: %[[E:[^ ]*]] = getelementptr i8, i8* %.hwasan.shadow, i64 %4
; RECOVER-ZERO-BASED-SHADOW: %[[E:[^ ]*]] = inttoptr i64 %[[D]] to i8*
; RECOVER: %[[MEMTAG:[^ ]*]] = load i8, i8* %[[E]]
; RECOVER: %[[F:[^ ]*]] = icmp ne i8 %[[PTRTAG]], %[[MEMTAG]]
; RECOVER: br i1 %[[F]], label {{.*}}, label {{.*}}, !prof {{.*}}

; RECOVER: call void asm sideeffect "brk #2340", "{x0}"(i64 %[[A]])
; RECOVER: br label

; ABORT: %[[A:[^ ]*]] = bitcast i128* %a to i8*
; ABORT: call void @llvm.hwasan.check.memaccess.shortgranules(i8* %.hwasan.shadow, i8* %[[A]], i32 4)

; CHECK: %[[G:[^ ]*]] = load i128, i128* %a, align 16
; CHECK: ret i128 %[[G]]

entry:
  %b = load i128, i128* %a, align 16
  ret i128 %b
}

define i40 @test_load40(i40* %a) sanitize_hwaddress {
; CHECK-LABEL: @test_load40(
; CHECK: %[[A:[^ ]*]] = ptrtoint i40* %a to i64
; ABORT: call void @__hwasan_loadN(i64 %[[A]], i64 5)
; RECOVER: call void @__hwasan_loadN_noabort(i64 %[[A]], i64 5)
; CHECK: %[[B:[^ ]*]] = load i40, i40* %a
; CHECK: ret i40 %[[B]]

entry:
  %b = load i40, i40* %a, align 4
  ret i40 %b
}

define void @test_store8(i8* %a, i8 %b) sanitize_hwaddress {
; CHECK-LABEL: @test_store8(
; RECOVER: %[[A:[^ ]*]] = ptrtoint i8* %a to i64
; RECOVER: %[[B:[^ ]*]] = lshr i64 %[[A]], 56
; RECOVER: %[[PTRTAG:[^ ]*]] = trunc i64 %[[B]] to i8
; RECOVER: %[[C:[^ ]*]] = and i64 %[[A]], 72057594037927935
; RECOVER: %[[D:[^ ]*]] = lshr i64 %[[C]], 4
; RECOVER-DYNAMIC-SHADOW: %[[E:[^ ]*]] = getelementptr i8, i8* %.hwasan.shadow, i64 %4
; RECOVER-ZERO-BASED-SHADOW: %[[E:[^ ]*]] = inttoptr i64 %[[D]] to i8*
; RECOVER: %[[MEMTAG:[^ ]*]] = load i8, i8* %[[E]]
; RECOVER: %[[F:[^ ]*]] = icmp ne i8 %[[PTRTAG]], %[[MEMTAG]]
; RECOVER: br i1 %[[F]], label {{.*}}, label {{.*}}, !prof {{.*}}

; RECOVER: call void asm sideeffect "brk #2352", "{x0}"(i64 %[[A]])
; RECOVER: br label

; ABORT: call void @llvm.hwasan.check.memaccess.shortgranules(i8* %.hwasan.shadow, i8* %a, i32 16)

; CHECK: store i8 %b, i8* %a, align 4
; CHECK: ret void

entry:
  store i8 %b, i8* %a, align 4
  ret void
}

define void @test_store16(i16* %a, i16 %b) sanitize_hwaddress {
; CHECK-LABEL: @test_store16(
; RECOVER: %[[A:[^ ]*]] = ptrtoint i16* %a to i64
; RECOVER: %[[B:[^ ]*]] = lshr i64 %[[A]], 56
; RECOVER: %[[PTRTAG:[^ ]*]] = trunc i64 %[[B]] to i8
; RECOVER: %[[C:[^ ]*]] = and i64 %[[A]], 72057594037927935
; RECOVER: %[[D:[^ ]*]] = lshr i64 %[[C]], 4
; RECOVER-DYNAMIC-SHADOW: %[[E:[^ ]*]] = getelementptr i8, i8* %.hwasan.shadow, i64 %4
; RECOVER-ZERO-BASED-SHADOW: %[[E:[^ ]*]] = inttoptr i64 %[[D]] to i8*
; RECOVER: %[[MEMTAG:[^ ]*]] = load i8, i8* %[[E]]
; RECOVER: %[[F:[^ ]*]] = icmp ne i8 %[[PTRTAG]], %[[MEMTAG]]
; RECOVER: br i1 %[[F]], label {{.*}}, label {{.*}}, !prof {{.*}}

; RECOVER: call void asm sideeffect "brk #2353", "{x0}"(i64 %[[A]])
; RECOVER: br label

; ABORT: %[[A:[^ ]*]] = bitcast i16* %a to i8*
; ABORT: call void @llvm.hwasan.check.memaccess.shortgranules(i8* %.hwasan.shadow, i8* %[[A]], i32 17)

; CHECK: store i16 %b, i16* %a, align 4
; CHECK: ret void

entry:
  store i16 %b, i16* %a, align 4
  ret void
}

define void @test_store32(i32* %a, i32 %b) sanitize_hwaddress {
; CHECK-LABEL: @test_store32(
; RECOVER: %[[A:[^ ]*]] = ptrtoint i32* %a to i64
; RECOVER: %[[B:[^ ]*]] = lshr i64 %[[A]], 56
; RECOVER: %[[PTRTAG:[^ ]*]] = trunc i64 %[[B]] to i8
; RECOVER: %[[C:[^ ]*]] = and i64 %[[A]], 72057594037927935
; RECOVER: %[[D:[^ ]*]] = lshr i64 %[[C]], 4
; RECOVER-DYNAMIC-SHADOW: %[[E:[^ ]*]] = getelementptr i8, i8* %.hwasan.shadow, i64 %4
; RECOVER-ZERO-BASED-SHADOW: %[[E:[^ ]*]] = inttoptr i64 %[[D]] to i8*
; RECOVER: %[[MEMTAG:[^ ]*]] = load i8, i8* %[[E]]
; RECOVER: %[[F:[^ ]*]] = icmp ne i8 %[[PTRTAG]], %[[MEMTAG]]
; RECOVER: br i1 %[[F]], label {{.*}}, label {{.*}}, !prof {{.*}}

; RECOVER: call void asm sideeffect "brk #2354", "{x0}"(i64 %[[A]])
; RECOVER: br label

; ABORT: %[[A:[^ ]*]] = bitcast i32* %a to i8*
; ABORT: call void @llvm.hwasan.check.memaccess.shortgranules(i8* %.hwasan.shadow, i8* %[[A]], i32 18)

; CHECK: store i32 %b, i32* %a, align 4
; CHECK: ret void

entry:
  store i32 %b, i32* %a, align 4
  ret void
}

define void @test_store64(i64* %a, i64 %b) sanitize_hwaddress {
; CHECK-LABEL: @test_store64(
; RECOVER: %[[A:[^ ]*]] = ptrtoint i64* %a to i64
; RECOVER: %[[B:[^ ]*]] = lshr i64 %[[A]], 56
; RECOVER: %[[PTRTAG:[^ ]*]] = trunc i64 %[[B]] to i8
; RECOVER: %[[C:[^ ]*]] = and i64 %[[A]], 72057594037927935
; RECOVER: %[[D:[^ ]*]] = lshr i64 %[[C]], 4
; RECOVER-DYNAMIC-SHADOW: %[[E:[^ ]*]] = getelementptr i8, i8* %.hwasan.shadow, i64 %4
; RECOVER-ZERO-BASED-SHADOW: %[[E:[^ ]*]] = inttoptr i64 %[[D]] to i8*
; RECOVER: %[[MEMTAG:[^ ]*]] = load i8, i8* %[[E]]
; RECOVER: %[[F:[^ ]*]] = icmp ne i8 %[[PTRTAG]], %[[MEMTAG]]
; RECOVER: br i1 %[[F]], label {{.*}}, label {{.*}}, !prof {{.*}}

; RECOVER: call void asm sideeffect "brk #2355", "{x0}"(i64 %[[A]])
; RECOVER: br label

; ABORT: %[[A:[^ ]*]] = bitcast i64* %a to i8*
; ABORT: call void @llvm.hwasan.check.memaccess.shortgranules(i8* %.hwasan.shadow, i8* %[[A]], i32 19)

; CHECK: store i64 %b, i64* %a, align 8
; CHECK: ret void

entry:
  store i64 %b, i64* %a, align 8
  ret void
}

define void @test_store128(i128* %a, i128 %b) sanitize_hwaddress {
; CHECK-LABEL: @test_store128(
; RECOVER: %[[A:[^ ]*]] = ptrtoint i128* %a to i64
; RECOVER: %[[B:[^ ]*]] = lshr i64 %[[A]], 56
; RECOVER: %[[PTRTAG:[^ ]*]] = trunc i64 %[[B]] to i8
; RECOVER: %[[C:[^ ]*]] = and i64 %[[A]], 72057594037927935
; RECOVER: %[[D:[^ ]*]] = lshr i64 %[[C]], 4
; RECOVER-DYNAMIC-SHADOW: %[[E:[^ ]*]] = getelementptr i8, i8* %.hwasan.shadow, i64 %4
; RECOVER-ZERO-BASED-SHADOW: %[[E:[^ ]*]] = inttoptr i64 %[[D]] to i8*
; RECOVER: %[[MEMTAG:[^ ]*]] = load i8, i8* %[[E]]
; RECOVER: %[[F:[^ ]*]] = icmp ne i8 %[[PTRTAG]], %[[MEMTAG]]
; RECOVER: br i1 %[[F]], label {{.*}}, label {{.*}}, !prof {{.*}}

; RECOVER: call void asm sideeffect "brk #2356", "{x0}"(i64 %[[A]])
; RECOVER: br label

; ABORT: %[[A:[^ ]*]] = bitcast i128* %a to i8*
; ABORT: call void @llvm.hwasan.check.memaccess.shortgranules(i8* %.hwasan.shadow, i8* %[[A]], i32 20)

; CHECK: store i128 %b, i128* %a, align 16
; CHECK: ret void

entry:
  store i128 %b, i128* %a, align 16
  ret void
}

define void @test_store40(i40* %a, i40 %b) sanitize_hwaddress {
; CHECK-LABEL: @test_store40(
; CHECK: %[[A:[^ ]*]] = ptrtoint i40* %a to i64
; ABORT: call void @__hwasan_storeN(i64 %[[A]], i64 5)
; RECOVER: call void @__hwasan_storeN_noabort(i64 %[[A]], i64 5)
; CHECK: store i40 %b, i40* %a
; CHECK: ret void

entry:
  store i40 %b, i40* %a, align 4
  ret void
}

define void @test_store_unaligned(i64* %a, i64 %b) sanitize_hwaddress {
; CHECK-LABEL: @test_store_unaligned(
; CHECK: %[[A:[^ ]*]] = ptrtoint i64* %a to i64
; ABORT: call void @__hwasan_storeN(i64 %[[A]], i64 8)
; RECOVER: call void @__hwasan_storeN_noabort(i64 %[[A]], i64 8)
; CHECK: store i64 %b, i64* %a, align 4
; CHECK: ret void

entry:
  store i64 %b, i64* %a, align 4
  ret void
}

define i8 @test_load_noattr(i8* %a) {
; CHECK-LABEL: @test_load_noattr(
; CHECK-NEXT: entry:
; CHECK-NEXT: %[[B:[^ ]*]] = load i8, i8* %a
; CHECK-NEXT: ret i8 %[[B]]

entry:
  %b = load i8, i8* %a, align 4
  ret i8 %b
}

define i8 @test_load_notmyattr(i8* %a) sanitize_address {
; CHECK-LABEL: @test_load_notmyattr(
; CHECK-NEXT: entry:
; CHECK-NEXT: %[[B:[^ ]*]] = load i8, i8* %a
; CHECK-NEXT: ret i8 %[[B]]

entry:
  %b = load i8, i8* %a, align 4
  ret i8 %b
}

define i8 @test_load_addrspace(i8 addrspace(256)* %a) sanitize_hwaddress {
; CHECK-LABEL: @test_load_addrspace(
; CHECK-NEXT: entry:
; CHECK-NEXT: %[[B:[^ ]*]] = load i8, i8 addrspace(256)* %a
; CHECK-NEXT: ret i8 %[[B]]

entry:
  %b = load i8, i8 addrspace(256)* %a, align 4
  ret i8 %b
}

; CHECK: declare void @__hwasan_init()

; CHECK:      define internal void @hwasan.module_ctor() comdat {
; CHECK-NEXT:   call void @__hwasan_init()
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
