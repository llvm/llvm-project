; RUN: llc -verify-machineinstrs -mtriple="powerpc64le-unknown-linux-gnu" -relocation-model=pic < %s | FileCheck %s

@a = thread_local global ptr null, align 8

define void @test_foo(ptr nocapture %x01, ptr nocapture %x02, ptr nocapture %x03, ptr nocapture %x04, ptr nocapture %x05, ptr nocapture %x06, ptr nocapture %x07, ptr nocapture %x08) #0 {
entry:

; CHECK-LABEL: test_foo:
; CHECK-DAG: stdu 1, {{-?[0-9]+}}(1)
; CHECK-DAG: mr [[BACKUP_3:[0-9]+]], 3
; CHECK-DAG: mr [[BACKUP_4:[0-9]+]], 4
; CHECK-DAG: mr [[BACKUP_5:[0-9]+]], 5
; CHECK-DAG: mr [[BACKUP_6:[0-9]+]], 6
; CHECK-DAG: mr [[BACKUP_7:[0-9]+]], 7
; CHECK-DAG: mr [[BACKUP_8:[0-9]+]], 8
; CHECK-DAG: mr [[BACKUP_9:[0-9]+]], 9
; CHECK-DAG: mr [[BACKUP_10:[0-9]+]], 10
; CHECK-DAG: std [[BACKUP_3]], {{-?[0-9]+}}(1)
; CHECK-DAG: std [[BACKUP_4]], {{-?[0-9]+}}(1)
; CHECK-DAG: std [[BACKUP_5]], {{-?[0-9]+}}(1)
; CHECK-DAG: std [[BACKUP_6]], {{-?[0-9]+}}(1)
; CHECK-DAG: std [[BACKUP_7]], {{-?[0-9]+}}(1)
; CHECK-DAG: std [[BACKUP_8]], {{-?[0-9]+}}(1)
; CHECK-DAG: std [[BACKUP_9]], {{-?[0-9]+}}(1)
; CHECK-DAG: std [[BACKUP_10]], {{-?[0-9]+}}(1)
; CHECK: bl __tls_get_addr
; CHECK-DAG: stw 3, 0([[BACKUP_3]])
; CHECK-DAG: stw 3, 0([[BACKUP_4]])
; CHECK-DAG: stw 3, 0([[BACKUP_5]])
; CHECK-DAG: stw 3, 0([[BACKUP_6]])
; CHECK-DAG: stw 3, 0([[BACKUP_7]])
; CHECK-DAG: stw 3, 0([[BACKUP_8]])
; CHECK-DAG: stw 3, 0([[BACKUP_9]])
; CHECK-DAG: stw 3, 0([[BACKUP_10]])
; CHECK: blr

  %0 = load ptr, ptr @a, align 8
  %cmp = icmp eq ptr %0, null
  br i1 %cmp, label %return, label %if.end

if.end:                                           ; preds = %entry
  store i32 0, ptr %x01, align 4
  store i32 0, ptr %x02, align 4
  store i32 0, ptr %x03, align 4
  store i32 0, ptr %x04, align 4
  store i32 0, ptr %x05, align 4
  store i32 0, ptr %x06, align 4
  store i32 0, ptr %x07, align 4
  store i32 0, ptr %x08, align 4
  br label %return

return:                                           ; preds = %entry, %if.end
  ret void
}
