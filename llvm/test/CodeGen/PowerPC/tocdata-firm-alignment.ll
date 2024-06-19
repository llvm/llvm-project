; RUN: opt -S -passes='default<O3>' < %s | FileCheck %s

target datalayout = "E-m:a-p:32:32-Fi32-i64:64-n32"
target triple = "powerpc-ibm-aix7.2.0.0"

%struct.widget = type { i8, i8, i8 }

; CHECK: @global = {{.*}}constant %struct.widget { i8 4, i8 0, i8 0 }, align 4 #0
@global = constant %struct.widget { i8 4, i8 0, i8 0 }, align 4 #0

define void @baz() #1 {
bb:
  call void @snork(ptr @global)
  ret void
}

define void @snork(ptr byval(%struct.widget) %arg) #1 {
bb:
  %load = load volatile ptr, ptr null, align 4
  ret void
}

attributes #0 = { "toc-data" }
attributes #1 = { "target-cpu"="pwr7" "target-features"="+altivec,+bpermd,+extdiv,+isa-v206-instructions,+vsx,-aix-shared-lib-tls-model-opt,-aix-small-local-dynamic-tls,-aix-small-local-exec-tls,-crbits,-crypto,-direct-move,-htm,-isa-v207-instructions,-isa-v30-instructions,-power8-vector,-power9-vector,-privileged,-quadword-atomics,-rop-protect,-spe" }
