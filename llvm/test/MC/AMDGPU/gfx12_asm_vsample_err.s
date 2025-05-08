// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1200 -show-encoding %s 2>&1 | FileCheck --check-prefix=GFX12-ERR --strict-whitespace %s

image_sample v[29:30], [v31, v32, v33], s[32:39], s[68:71] dmask:0x5 dim:SQ_RSRC_IMG_3D th:TH_LOAD_BYPASS scope:SCOPE_SYS cfs:CFS_128B
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: Cache fill size is not supported on this GPU
// GFX12-ERR-NEXT:{{^}}image_sample v[29:30], [v31, v32, v33], s[32:39], s[68:71] dmask:0x5 dim:SQ_RSRC_IMG_3D th:TH_LOAD_BYPASS scope:SCOPE_SYS cfs:CFS_128B
// GFX12-ERR-NEXT:{{^}}                                                                                                                              ^

image_sample v[29:30], [v31, v32, v33], s[32:39], s[68:71] dmask:0x5 dim:SQ_RSRC_IMG_3D th:TH_LOAD_BYPASS scope:SCOPE_SYS cfs:CFS_64B
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: Cache fill size is not supported on this GPU
// GFX12-ERR-NEXT:{{^}}image_sample v[29:30], [v31, v32, v33], s[32:39], s[68:71] dmask:0x5 dim:SQ_RSRC_IMG_3D th:TH_LOAD_BYPASS scope:SCOPE_SYS cfs:CFS_64B
// GFX12-ERR-NEXT:{{^}}                                                                                                                              ^

image_sample v[29:30], [v31, v32, v33], s[32:39], s[68:71] dmask:0x5 dim:SQ_RSRC_IMG_3D th:TH_LOAD_BYPASS scope:SCOPE_SYS cfs:CFS_32B
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: Cache fill size is not supported on this GPU
// GFX12-ERR-NEXT:{{^}}image_sample v[29:30], [v31, v32, v33], s[32:39], s[68:71] dmask:0x5 dim:SQ_RSRC_IMG_3D th:TH_LOAD_BYPASS scope:SCOPE_SYS cfs:CFS_32B
// GFX12-ERR-NEXT:{{^}}                                                                                                                              ^

image_gather4 v[18:21], [v22, v23], s[24:31], s[88:91] dmask:0x4 dim:SQ_RSRC_IMG_2D th:TH_LOAD_BYPASS scope:SCOPE_SYS cfs:CFS_128B
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: Cache fill size is not supported on this GPU
// GFX12-ERR-NEXT:{{^}}image_gather4 v[18:21], [v22, v23], s[24:31], s[88:91] dmask:0x4 dim:SQ_RSRC_IMG_2D th:TH_LOAD_BYPASS scope:SCOPE_SYS cfs:CFS_128B
// GFX12-ERR-NEXT:{{^}}                                                                                                                          ^

image_gather4 v[18:21], [v22, v23], s[24:31], s[88:91] dmask:0x4 dim:SQ_RSRC_IMG_2D th:TH_LOAD_BYPASS scope:SCOPE_SYS cfs:CFS_64B
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: Cache fill size is not supported on this GPU
// GFX12-ERR-NEXT:{{^}}image_gather4 v[18:21], [v22, v23], s[24:31], s[88:91] dmask:0x4 dim:SQ_RSRC_IMG_2D th:TH_LOAD_BYPASS scope:SCOPE_SYS cfs:CFS_64B
// GFX12-ERR-NEXT:{{^}}                                                                                                                          ^

image_gather4 v[18:21], [v22, v23], s[24:31], s[88:91] dmask:0x4 dim:SQ_RSRC_IMG_2D th:TH_LOAD_BYPASS scope:SCOPE_SYS cfs:CFS_32B
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: Cache fill size is not supported on this GPU
// GFX12-ERR-NEXT:{{^}}image_gather4 v[18:21], [v22, v23], s[24:31], s[88:91] dmask:0x4 dim:SQ_RSRC_IMG_2D th:TH_LOAD_BYPASS scope:SCOPE_SYS cfs:CFS_32B
// GFX12-ERR-NEXT:{{^}}                                                                                                                          ^

image_get_lod v[64:67], [v32, v33, v34], s[4:11], s[100:103] dmask:0xf dim:SQ_RSRC_IMG_2D_ARRAY cfs:CFS_128B
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: Cache fill size is not supported on this GPU
// GFX12-ERR-NEXT:{{^}}image_get_lod v[64:67], [v32, v33, v34], s[4:11], s[100:103] dmask:0xf dim:SQ_RSRC_IMG_2D_ARRAY cfs:CFS_128B
// GFX12-ERR-NEXT:{{^}}                                                                                                    ^

image_get_lod v[64:67], [v32, v33, v34], s[4:11], s[100:103] dmask:0xf dim:SQ_RSRC_IMG_2D_ARRAY cfs:CFS_64B
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: Cache fill size is not supported on this GPU
// GFX12-ERR-NEXT:{{^}}image_get_lod v[64:67], [v32, v33, v34], s[4:11], s[100:103] dmask:0xf dim:SQ_RSRC_IMG_2D_ARRAY cfs:CFS_64B
// GFX12-ERR-NEXT:{{^}}                                                                                                    ^

image_get_lod v[64:67], [v32, v33, v34], s[4:11], s[100:103] dmask:0xf dim:SQ_RSRC_IMG_2D_ARRAY cfs:CFS_32B
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: Cache fill size is not supported on this GPU
// GFX12-ERR-NEXT:{{^}}image_get_lod v[64:67], [v32, v33, v34], s[4:11], s[100:103] dmask:0xf dim:SQ_RSRC_IMG_2D_ARRAY cfs:CFS_32B
// GFX12-ERR-NEXT:{{^}}                                                                                                    ^
