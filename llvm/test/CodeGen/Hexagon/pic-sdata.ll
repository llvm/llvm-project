; RUN: llc -mtriple=hexagon-unknown-elf -hexagon-small-data-threshold=8 -relocation-model=static < %s | FileCheck --check-prefixes=CHECK,STATIC %s
; RUN: llc -mtriple=hexagon-unknown-elf -hexagon-small-data-threshold=8 -relocation-model=pic < %s | FileCheck --check-prefixes=CHECK,PIC %s

; If a global has a specified section, it should probably be placed in that
; section, but with PIC any accesses to globals in small data should still
; go through GOT.

@g0 = global i32 zeroinitializer
@g1 = global i32 zeroinitializer, section ".sdata"

; CHECK-LABEL: f0:
; STATIC: memw(gp+#g0)
; PIC: r[[R0:[0-9]+]] = add(pc,##_GLOBAL_OFFSET_TABLE_@PCREL)
; PIC: = memw(r[[R0]]+##g0@GOT)
define i32 @f0() #0 {
  %v0 = load i32, ptr @g0
  ret i32 %v0
}

; CHECK-LABEL: f1:
; STATIC: memw(gp+#g1)
; PIC: r[[R1:[0-9]+]] = add(pc,##_GLOBAL_OFFSET_TABLE_@PCREL)
; PIC: = memw(r[[R1]]+##g1@GOT)
define i32 @f1() #0 {
  %v0 = load i32, ptr @g1
  ret i32 %v0
}

; CHECK-LABEL: f2:
; STATIC: CONST64(#123456789012345678)
; PIC: r0 = ##-1506741426
; PIC: r1 = ##28744523
define i64 @f2() #0 {
  ret i64 123456789012345678
}

attributes #0 = { nounwind "target-cpu"="hexagonv60" }
