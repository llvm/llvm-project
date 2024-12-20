// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1200 -show-encoding %s 2>&1 | FileCheck --check-prefix=GFX12-ERR --strict-whitespace %s

image_load v0, v0, s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_1D th:TH_LOAD_BYPASS scope:SCOPE_SYS cfs:CFS_128B
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: Cache fill size is not supported on this GPU
// GFX12-ERR-NEXT:{{^}}image_load v0, v0, s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_1D th:TH_LOAD_BYPASS scope:SCOPE_SYS cfs:CFS_128B
// GFX12-ERR-NEXT:{{^}}                                                                                             ^

image_load v0, v0, s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_1D th:TH_LOAD_BYPASS scope:SCOPE_SYS cfs:CFS_64B
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: Cache fill size is not supported on this GPU
// GFX12-ERR-NEXT:{{^}}image_load v0, v0, s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_1D th:TH_LOAD_BYPASS scope:SCOPE_SYS cfs:CFS_64B
// GFX12-ERR-NEXT:{{^}}                                                                                             ^

image_load v0, v0, s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_1D th:TH_LOAD_BYPASS scope:SCOPE_SYS cfs:CFS_32B
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: Cache fill size is not supported on this GPU
// GFX12-ERR-NEXT:{{^}}image_load v0, v0, s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_1D th:TH_LOAD_BYPASS scope:SCOPE_SYS cfs:CFS_32B
// GFX12-ERR-NEXT:{{^}}                                                                                             ^

image_load_pck v5, v1, s[8:15] dmask:0x1 dim:SQ_RSRC_IMG_1D th:TH_LOAD_NT cfs:CFS_128B
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: Cache fill size is not supported on this GPU
// GFX12-ERR-NEXT:{{^}}image_load_pck v5, v1, s[8:15] dmask:0x1 dim:SQ_RSRC_IMG_1D th:TH_LOAD_NT cfs:CFS_128B
// GFX12-ERR-NEXT:{{^}}                                                                              ^

image_load_pck v5, v1, s[8:15] dmask:0x1 dim:SQ_RSRC_IMG_1D th:TH_LOAD_NT cfs:CFS_64B
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: Cache fill size is not supported on this GPU
// GFX12-ERR-NEXT:{{^}}image_load_pck v5, v1, s[8:15] dmask:0x1 dim:SQ_RSRC_IMG_1D th:TH_LOAD_NT cfs:CFS_64B
// GFX12-ERR-NEXT:{{^}}                                                                              ^

image_load_pck v5, v1, s[8:15] dmask:0x1 dim:SQ_RSRC_IMG_1D th:TH_LOAD_NT cfs:CFS_32B
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: Cache fill size is not supported on this GPU
// GFX12-ERR-NEXT:{{^}}image_load_pck v5, v1, s[8:15] dmask:0x1 dim:SQ_RSRC_IMG_1D th:TH_LOAD_NT cfs:CFS_32B
// GFX12-ERR-NEXT:{{^}}                                                                              ^

image_store v0, v0, s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_1D th:TH_STORE_BYPASS scope:SCOPE_SYS cfs:CFS_128B
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: Cache fill size is not supported on this GPU
// GFX12-ERR-NEXT:{{^}}image_store v0, v0, s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_1D th:TH_STORE_BYPASS scope:SCOPE_SYS cfs:CFS_128B
// GFX12-ERR-NEXT:{{^}}                                                                                               ^

image_store v0, v0, s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_1D th:TH_STORE_BYPASS scope:SCOPE_SYS cfs:CFS_64B
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: Cache fill size is not supported on this GPU
// GFX12-ERR-NEXT:{{^}}image_store v0, v0, s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_1D th:TH_STORE_BYPASS scope:SCOPE_SYS cfs:CFS_64B
// GFX12-ERR-NEXT:{{^}}                                                                                               ^

image_store v0, v0, s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_1D th:TH_STORE_BYPASS scope:SCOPE_SYS cfs:CFS_32B
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: Cache fill size is not supported on this GPU
// GFX12-ERR-NEXT:{{^}}image_store v0, v0, s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_1D th:TH_STORE_BYPASS scope:SCOPE_SYS cfs:CFS_32B
// GFX12-ERR-NEXT:{{^}}                                                                                               ^

image_store_pck v5, v1, s[8:15] dmask:0x1 dim:SQ_RSRC_IMG_1D th:TH_STORE_NT cfs:CFS_128B
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: Cache fill size is not supported on this GPU
// GFX12-ERR-NEXT:{{^}}image_store_pck v5, v1, s[8:15] dmask:0x1 dim:SQ_RSRC_IMG_1D th:TH_STORE_NT cfs:CFS_128B
// GFX12-ERR-NEXT:{{^}}                                                                                ^

image_store_pck v5, v1, s[8:15] dmask:0x1 dim:SQ_RSRC_IMG_1D th:TH_STORE_NT cfs:CFS_64B
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: Cache fill size is not supported on this GPU
// GFX12-ERR-NEXT:{{^}}image_store_pck v5, v1, s[8:15] dmask:0x1 dim:SQ_RSRC_IMG_1D th:TH_STORE_NT cfs:CFS_64B
// GFX12-ERR-NEXT:{{^}}                                                                                ^

image_store_pck v5, v1, s[8:15] dmask:0x1 dim:SQ_RSRC_IMG_1D th:TH_STORE_NT cfs:CFS_32B
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: Cache fill size is not supported on this GPU
// GFX12-ERR-NEXT:{{^}}image_store_pck v5, v1, s[8:15] dmask:0x1 dim:SQ_RSRC_IMG_1D th:TH_STORE_NT cfs:CFS_32B
// GFX12-ERR-NEXT:{{^}}                                                                                ^

image_atomic_and v0, v0, s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_1D th:TH_ATOMIC_NT cfs:CFS_128B
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: Cache fill size is not supported on this GPU
// GFX12-ERR-NEXT:{{^}}image_atomic_and v0, v0, s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_1D th:TH_ATOMIC_NT cfs:CFS_128B
// GFX12-ERR-NEXT:{{^}}                                                                                 ^

image_atomic_and v0, v0, s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_1D th:TH_ATOMIC_NT cfs:CFS_64B
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: Cache fill size is not supported on this GPU
// GFX12-ERR-NEXT:{{^}}image_atomic_and v0, v0, s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_1D th:TH_ATOMIC_NT cfs:CFS_64B
// GFX12-ERR-NEXT:{{^}}                                                                                 ^

image_atomic_and v0, v0, s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_1D th:TH_ATOMIC_NT cfs:CFS_32B
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: Cache fill size is not supported on this GPU
// GFX12-ERR-NEXT:{{^}}image_atomic_and v0, v0, s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_1D th:TH_ATOMIC_NT cfs:CFS_32B
// GFX12-ERR-NEXT:{{^}}                                                                                 ^

image_atomic_add_flt v0, v0, s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_1D th:TH_ATOMIC_NT cfs:CFS_128B
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: Cache fill size is not supported on this GPU
// GFX12-ERR-NEXT:{{^}}image_atomic_add_flt v0, v0, s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_1D th:TH_ATOMIC_NT cfs:CFS_128B
// GFX12-ERR-NEXT:{{^}}                                                                                     ^

image_atomic_add_flt v0, v0, s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_1D th:TH_ATOMIC_NT cfs:CFS_64B
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: Cache fill size is not supported on this GPU
// GFX12-ERR-NEXT:{{^}}image_atomic_add_flt v0, v0, s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_1D th:TH_ATOMIC_NT cfs:CFS_64B
// GFX12-ERR-NEXT:{{^}}                                                                                     ^

image_atomic_add_flt v0, v0, s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_1D th:TH_ATOMIC_NT cfs:CFS_32B
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: Cache fill size is not supported on this GPU
// GFX12-ERR-NEXT:{{^}}image_atomic_add_flt v0, v0, s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_1D th:TH_ATOMIC_NT cfs:CFS_32B
// GFX12-ERR-NEXT:{{^}}                                                                                     ^

image_get_resinfo v4, v32, s[96:103] dmask:0x1 dim:SQ_RSRC_IMG_1D th:TH_LOAD_BYPASS scope:SCOPE_SYS cfs:CFS_128B
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: Cache fill size is not supported on this GPU
// GFX12-ERR-NEXT:{{^}}image_get_resinfo v4, v32, s[96:103] dmask:0x1 dim:SQ_RSRC_IMG_1D th:TH_LOAD_BYPASS scope:SCOPE_SYS cfs:CFS_128B
// GFX12-ERR-NEXT:{{^}}                                                                                                        ^

image_get_resinfo v4, v32, s[96:103] dmask:0x1 dim:SQ_RSRC_IMG_1D th:TH_LOAD_BYPASS scope:SCOPE_SYS cfs:CFS_64B
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: Cache fill size is not supported on this GPU
// GFX12-ERR-NEXT:{{^}}image_get_resinfo v4, v32, s[96:103] dmask:0x1 dim:SQ_RSRC_IMG_1D th:TH_LOAD_BYPASS scope:SCOPE_SYS cfs:CFS_64B
// GFX12-ERR-NEXT:{{^}}                                                                                                        ^

image_get_resinfo v4, v32, s[96:103] dmask:0x1 dim:SQ_RSRC_IMG_1D th:TH_LOAD_BYPASS scope:SCOPE_SYS cfs:CFS_32B
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: Cache fill size is not supported on this GPU
// GFX12-ERR-NEXT:{{^}}image_get_resinfo v4, v32, s[96:103] dmask:0x1 dim:SQ_RSRC_IMG_1D th:TH_LOAD_BYPASS scope:SCOPE_SYS cfs:CFS_32B
// GFX12-ERR-NEXT:{{^}}                                                                                                        ^
