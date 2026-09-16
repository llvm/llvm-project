; RUN: llc -mtriple=bpfel -mcpu=v4 -filetype=asm -o - %s | FileCheck --check-prefix=ASM %s
; RUN: llc -mtriple=bpfel -mcpu=v4 -filetype=obj -o %t %s
; RUN: llvm-readelf -S %t | FileCheck --check-prefix=SECTIONS %s
; RUN: llvm-readelf -r %t | FileCheck --check-prefix=RELOCS %s

; Verify that invoke/landingpad with cleanup produces a .bpf_cleanup section
; containing (call_site, landing_pad) pairs with R_BPF_64_NODYLD32 relocations.
; Models a Rust BTreeMap insertion that may fail, with Drop cleanup
; calling bpf_free to release allocated nodes.

declare ptr @bpf_alloc(i64, i64)
declare void @bpf_free(ptr)
declare void @btree_insert(ptr, ptr, i64)
declare i32 @rust_personality(i32, i64, ptr, ptr)

; SECTIONS: .bpf_cleanup

define void @stat_inc(ptr %map, ptr %key, i64 %val) personality ptr @rust_personality {
entry:
  %node = call ptr @bpf_alloc(i64 256, i64 0)
  %null = icmp eq ptr %node, null
  br i1 %null, label %oom, label %insert

insert:
  invoke void @btree_insert(ptr %map, ptr %node, i64 %val)
          to label %done unwind label %cleanup

done:
  ret void

cleanup:
  %lp = landingpad { ptr, i32 } cleanup
  call void @bpf_free(ptr %node)
  resume { ptr, i32 } %lp

oom:
  ret void
}

; ASM-LABEL: stat_inc:
; ASM:         r1 = 256
; ASM:         call bpf_alloc
; ASM:         if r0 == 0 goto
;   Normal path: call btree_insert then exit
; ASM:       .Ltmp0:
; ASM:         call btree_insert
; ASM:       .Ltmp1:
; ASM:         exit
;   Cleanup landing pad: free the node, then resume unwinding
; ASM:       .Ltmp2:
; ASM:         r1 = r7
; ASM:         call bpf_free
; ASM:         call _Unwind_Resume
;   .bpf_cleanup entry: [.Ltmp0, .Ltmp1) -> .Ltmp2
; ASM:         .section .bpf_cleanup,"a",@progbits
; ASM-NEXT:    .long .Ltmp0
; ASM-NEXT:    .long .Ltmp1
; ASM-NEXT:    .long .Ltmp2

define void @two_allocs(ptr %map) personality ptr @rust_personality {
entry:
  %n1 = call ptr @bpf_alloc(i64 256, i64 0)
  invoke void @btree_insert(ptr %map, ptr %n1, i64 1)
          to label %second unwind label %clean1

second:
  %n2 = call ptr @bpf_alloc(i64 512, i64 0)
  invoke void @btree_insert(ptr %map, ptr %n2, i64 2)
          to label %done unwind label %clean2

done:
  ret void

clean2:
  ; Must free both n2 and n1
  %lp2 = landingpad { ptr, i32 } cleanup
  call void @bpf_free(ptr %n2)
  call void @bpf_free(ptr %n1)
  resume { ptr, i32 } %lp2

clean1:
  ; Only n1 allocated at this point
  %lp1 = landingpad { ptr, i32 } cleanup
  call void @bpf_free(ptr %n1)
  resume { ptr, i32 } %lp1
}

; ASM-LABEL: two_allocs:
;   First alloc + invoke
; ASM:         r1 = 256
; ASM:         call bpf_alloc
; ASM:       .Ltmp3:
; ASM:         call btree_insert
; ASM:       .Ltmp4:
;   Second alloc + invoke
; ASM:         r1 = 512
; ASM:         call bpf_alloc
; ASM:       .Ltmp6:
; ASM:         call btree_insert
; ASM:       .Ltmp7:
; ASM:         exit
;   clean2: free n2, then fall through to free n1
; ASM:       .Ltmp8:
; ASM:         call bpf_free
;   clean1 entry point, shared tail: free n1 then resume
; ASM:       .Ltmp5:
; ASM:         call bpf_free
; ASM:         call _Unwind_Resume
;   .bpf_cleanup entries: two triples
; ASM:         .section .bpf_cleanup,"a",@progbits
; ASM-NEXT:    .long .Ltmp3
; ASM-NEXT:    .long .Ltmp4
; ASM-NEXT:    .long .Ltmp5
; ASM-NEXT:    .long .Ltmp6
; ASM-NEXT:    .long .Ltmp7
; ASM-NEXT:    .long .Ltmp8

; Models std::panic::catch_unwind: catch the exception, free resources,
; and return normally. The landing pad uses "catch ptr null" (catch-all)
; instead of "cleanup", and does NOT call _Unwind_Resume — unwinding stops here.

define i64 @with_catch_unwind(ptr %map, i64 %val) personality ptr @rust_personality {
entry:
  %node = call ptr @bpf_alloc(i64 128, i64 0)
  invoke void @btree_insert(ptr %map, ptr %node, i64 %val)
          to label %ok unwind label %caught

ok:
  ret i64 0

caught:
  %lp = landingpad { ptr, i32 }
    catch ptr null
  call void @bpf_free(ptr %node)
  ret i64 1
}

; ASM-LABEL: with_catch_unwind:
; ASM:         r1 = 128
; ASM:         call bpf_alloc
; ASM:       .Ltmp9:
; ASM:         call btree_insert
; ASM:       .Ltmp10:
; ASM:         r0 = 0
; ASM:         exit
;   Catch block: free the node, return 1 (no _Unwind_Resume)
; ASM:       .Ltmp11:
; ASM:         call bpf_free
; ASM:         r0 = 1
; ASM:         exit
;   .bpf_cleanup entry for the catch
; ASM:         .section .bpf_cleanup,"a",@progbits
; ASM-NEXT:    .long .Ltmp9
; ASM-NEXT:    .long .Ltmp10
; ASM-NEXT:    .long .Ltmp11

; .bpf_cleanup should have R_BPF_64_NODYLD32 relocations
; 4 entries x 3 fields = 12 relocs
; RELOCS: .rel.bpf_cleanup
; RELOCS-COUNT-12: R_BPF_64_NODYLD32 {{.*}} .text
