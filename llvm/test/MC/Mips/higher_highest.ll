; DISABLE: llc -march=mips64el -mcpu=mips64 -mattr=n64  -force-mips-long-branch -filetype=obj < %s -o - | elf-dump --dump-section-data | FileCheck %s
; RUN: false
; XFAIL: *
; Disabled because currently we don't have a way to generate these relocations.
;
; Check that the R_MIPS_HIGHER and R_MIPS_HIGHEST relocations were created.

; CHECK:     ('r_type', 0x1d)
; CHECK:     ('r_type', 0x1d)
; CHECK:     ('r_type', 0x1c)
; CHECK:     ('r_type', 0x1c)

@g0 = external global i32

define void @foo1(i32 %s) nounwind {
entry:

  %tobool = icmp eq i32 %s, 0
  br i1 %tobool, label %if.end, label %if.then

if.then:                                          ; preds = %entry
  %0 = load i32* @g0, align 4
  %add = add nsw i32 %0, 12
  store i32 %add, i32* @g0, align 4
  br label %if.end

if.end:                                           ; preds = %entry, %if.then
  ret void
}

