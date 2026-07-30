; Verify that the .toc section is aligned on an 8-byte boundary.

; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc64-unknown-linux-gnu \
; RUN: -mcpu=pwr8 -filetype=obj -ppc-min-jump-table-entries=4 -o - | llvm-readobj --sections - | FileCheck %s

define void @test(ptr %a) {
entry:
  %a.addr = alloca ptr, align 8
  store ptr %a, ptr %a.addr, align 8
  %0 = load ptr,  ptr %a.addr, align 8
  %incdec.ptr = getelementptr inbounds i32, ptr %0, i32 1
  store ptr %incdec.ptr, ptr %a.addr, align 8
  %1 = load i32,  ptr %0, align 4
  switch i32 %1, label %sw.epilog [
    i32 17, label %sw.bb
    i32 13, label %sw.bb1
    i32 11, label %sw.bb2
    i32 7, label %sw.bb3
    i32 5, label %sw.bb4
    i32 3, label %sw.bb5
    i32 2, label %sw.bb6
  ]

sw.bb:                                            ; preds = %entry
  %2 = load ptr,  ptr %a.addr, align 8
  store i32 2, ptr %2, align 4
  br label %sw.epilog

sw.bb1:                                           ; preds = %entry
  %3 = load ptr,  ptr %a.addr, align 8
  store i32 3, ptr %3, align 4
  br label %sw.epilog

sw.bb2:                                           ; preds = %entry
  %4 = load ptr,  ptr %a.addr, align 8
  store i32 5, ptr %4, align 4
  br label %sw.epilog

sw.bb3:                                           ; preds = %entry
  %5 = load ptr,  ptr %a.addr, align 8
  store i32 7, ptr %5, align 4
  br label %sw.epilog

sw.bb4:                                           ; preds = %entry
  %6 = load ptr,  ptr %a.addr, align 8
  store i32 11, ptr %6, align 4
  br label %sw.epilog

sw.bb5:                                           ; preds = %entry
  %7 = load ptr,  ptr %a.addr, align 8
  store i32 13, ptr %7, align 4
  br label %sw.epilog

sw.bb6:                                           ; preds = %entry
  %8 = load ptr,  ptr %a.addr, align 8
  store i32 17, ptr %8, align 4
  br label %sw.epilog

sw.epilog:                                        ; preds = %entry, %sw.bb6, %sw.bb5, %sw.bb4, %sw.bb3, %sw.bb2, %sw.bb1, %sw.bb
  ret void
}

; CHECK: Name: .toc
; CHECK: AddressAlignment: 8
; CHECK: Name: .rela.toc

; This test was generated from the following from PR22711:

;void test(int *a) {
;  switch (*a++) {
;  case 17: *a =  2; break;
;  case 13: *a =  3; break;
;  case 11: *a =  5; break;
;  case  7: *a =  7; break;
;  case  5: *a = 11; break;
;  case  3: *a = 13; break;
;  case  2: *a = 17; break;
;  }
;}
