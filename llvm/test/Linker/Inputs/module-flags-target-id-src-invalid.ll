; This file is used with module-flags-target-id-dst-*.ll

; Invalid target id: feature must ends with +/-.
!llvm.module.flags = !{ !0 }
!0 = !{ i32 8, !"target-id", !"amdgcn-amd-amdhsa--gfx908:xnack" }
