; RUN: llc --relocation-model=pic  \
; RUN:   -mtriple=ppc32 < %s | FileCheck  %s 

@g = global i32 10, align 4

; Function Attrs: noinline nounwind optnone uwtable
define i32 @main() #0 {
; CHECK-LABEL: main:
; CHECK-NOT: evstdd
entry:
  %retval = alloca i32, align 4
  store i32 0, ptr %retval, align 4
  %0 = load i32, ptr @g, align 4
  ret i32 %0
}

attributes #0 = { noinline nounwind optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="e500" "target-features"="+spe,-altivec,-bpermd,-crbits,-crypto,-direct-move,-extdiv,-htm,-isa-v206-instructions,-isa-v207-instructions,-isa-v30-instructions,-power8-vector,-power9-vector,-privileged,-quadword-atomics,-rop-protect,-vsx" }

