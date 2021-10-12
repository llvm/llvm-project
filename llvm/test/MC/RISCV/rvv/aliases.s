# RUN: llvm-mc --triple=riscv64 -mattr +experimental-v < %s --show-encoding 2>&1 \
# RUN:   -mattr +d | FileCheck --check-prefix=ALIAS %s
# RUN: llvm-mc --triple=riscv64 -mattr=+experimental-v --riscv-no-aliases < %s \
# RUN:   -mattr +d --show-encoding 2>&1 | FileCheck --check-prefix=NO-ALIAS %s

# ALIAS:    vwcvt.x.x.v     v2, v1, v0.t    # encoding: [0x57,0x61,0x10,0xc4]
# NO-ALIAS: vwadd.vx        v2, v1, zero, v0.t # encoding: [0x57,0x61,0x10,0xc4]
vwcvt.x.x.v v2, v1, v0.t
# ALIAS:    vwcvtu.x.x.v    v2, v1, v0.t    # encoding: [0x57,0x61,0x10,0xc0]
# NO-ALIAS: vwaddu.vx       v2, v1, zero, v0.t # encoding: [0x57,0x61,0x10,0xc0]
vwcvtu.x.x.v v2, v1, v0.t
# ALIAS:    vnot.v  v2, v2, v0.t            # encoding: [0x57,0xb1,0x2f,0x2c]
# NO-ALIAS: vxor.vi v2, v2, -1, v0.t        # encoding: [0x57,0xb1,0x2f,0x2c]
vnot.v v2, v2, v0.t
# ALIAS:    vmsltu.vv       v2, v1, v2, v0.t # encoding: [0x57,0x01,0x11,0x68]
# NO-ALIAS: vmsltu.vv       v2, v1, v2, v0.t # encoding: [0x57,0x01,0x11,0x68]
vmsgtu.vv v2, v2, v1, v0.t
# ALIAS:    vmslt.vv        v2, v1, v2, v0.t # encoding: [0x57,0x01,0x11,0x6c]
# NO-ALIAS: vmslt.vv        v2, v1, v2, v0.t # encoding: [0x57,0x01,0x11,0x6c]
vmsgt.vv v2, v2, v1, v0.t
# ALIAS:    vmsleu.vv       v2, v1, v2, v0.t # encoding: [0x57,0x01,0x11,0x70]
# NO-ALIAS: vmsleu.vv       v2, v1, v2, v0.t # encoding: [0x57,0x01,0x11,0x70]
vmsgeu.vv v2, v2, v1, v0.t
# ALIAS:    vmsle.vv        v2, v1, v2, v0.t # encoding: [0x57,0x01,0x11,0x74]
# NO-ALIAS: vmsle.vv        v2, v1, v2, v0.t # encoding: [0x57,0x01,0x11,0x74]
vmsge.vv v2, v2, v1, v0.t
# ALIAS:    vmsleu.vi       v2, v2, 15, v0.t # encoding: [0x57,0xb1,0x27,0x70]
# NO-ALIAS: vmsleu.vi       v2, v2, 15, v0.t # encoding: [0x57,0xb1,0x27,0x70]
vmsltu.vi v2, v2, 16, v0.t
# ALIAS:    vmsle.vi        v2, v2, 15, v0.t # encoding: [0x57,0xb1,0x27,0x74]
# NO-ALIAS: vmsle.vi        v2, v2, 15, v0.t # encoding: [0x57,0xb1,0x27,0x74]
vmslt.vi v2, v2, 16, v0.t
# ALIAS:    vmsgtu.vi       v2, v2, 15, v0.t # encoding: [0x57,0xb1,0x27,0x78]
# NO-ALIAS: vmsgtu.vi       v2, v2, 15, v0.t # encoding: [0x57,0xb1,0x27,0x78]
vmsgeu.vi v2, v2, 16, v0.t
# ALIAS:    vmsgt.vi        v2, v2, 15, v0.t # encoding: [0x57,0xb1,0x27,0x7c]
# NO-ALIAS: vmsgt.vi        v2, v2, 15, v0.t # encoding: [0x57,0xb1,0x27,0x7c]
vmsge.vi v2, v2, 16, v0.t
# ALIAS:    vmflt.vv        v2, v1, v2, v0.t # encoding: [0x57,0x11,0x11,0x6c]
# NO-ALIAS: vmflt.vv        v2, v1, v2, v0.t # encoding: [0x57,0x11,0x11,0x6c]
vmfgt.vv v2, v2, v1, v0.t
# ALIAS:    vmfle.vv        v2, v1, v2, v0.t # encoding: [0x57,0x11,0x11,0x64]
# NO-ALIAS: vmfle.vv        v2, v1, v2, v0.t # encoding: [0x57,0x11,0x11,0x64]
vmfge.vv v2, v2, v1, v0.t
# ALIAS:    vmmv.m v0, v1                  # encoding: [0x57,0xa0,0x10,0x66]
# NO-ALIAS: vmand.mm        v0, v1, v1      # encoding: [0x57,0xa0,0x10,0x66]
vmmv.m v0, v1
# ALIAS:    vmclr.m v0                      # encoding: [0x57,0x20,0x00,0x6e]
# NO-ALIAS: vmxor.mm        v0, v0, v0      # encoding: [0x57,0x20,0x00,0x6e]
vmclr.m v0
# ALIAS:    vmset.m v0                      # encoding: [0x57,0x20,0x00,0x7e]
# NO-ALIAS: vmxnor.mm       v0, v0, v0      # encoding: [0x57,0x20,0x00,0x7e]
vmset.m v0
# ALIAS:    vmnot.m v0, v1                  # encoding: [0x57,0xa0,0x10,0x76]
# NO-ALIAS: vmnand.mm       v0, v1, v1      # encoding: [0x57,0xa0,0x10,0x76]
vmnot.m v0, v1
# ALIAS:    vl1r.v          v0, (a0)        # encoding: [0x07,0x00,0x85,0x02]
# NO-ALIAS: vl1re8.v        v0, (a0)        # encoding: [0x07,0x00,0x85,0x02]
vl1r.v v0, (a0) 
# ALIAS:    vl2r.v          v0, (a0)        # encoding: [0x07,0x00,0x85,0x22]
# NO-ALIAS: vl2re8.v        v0, (a0)        # encoding: [0x07,0x00,0x85,0x22]
vl2r.v v0, (a0) 
# ALIAS:    vl4r.v          v0, (a0)        # encoding: [0x07,0x00,0x85,0x62]
# NO-ALIAS: vl4re8.v        v0, (a0)        # encoding: [0x07,0x00,0x85,0x62]
vl4r.v v0, (a0) 
# ALIAS:    vl8r.v          v0, (a0)        # encoding: [0x07,0x00,0x85,0xe2]
# NO-ALIAS: vl8re8.v        v0, (a0)        # encoding: [0x07,0x00,0x85,0xe2]
vl8r.v v0, (a0) 
# ALIAS:    vneg.v          v2, v1, v0.t    # encoding: [0x57,0x41,0x10,0x0c]
# NO-ALIAS: vrsub.vx        v2, v1, zero, v0.t # encoding: [0x57,0x41,0x10,0x0c]
vneg.v v2, v1, v0.t 
# ALIAS:    vncvt.x.x.w     v2, v1, v0.t    # encoding: [0x57,0x41,0x10,0xb0]
# NO-ALIAS: vnsrl.wx        v2, v1, zero, v0.t # encoding: [0x57,0x41,0x10,0xb0]
vncvt.x.x.w v2, v1, v0.t 
# ALIAS:    vfneg.v         v2, v1, v0.t     # encoding: [0x57,0x91,0x10,0x24]
# NO-ALIAS: vfsgnjn.vv      v2, v1, v1, v0.t # encoding: [0x57,0x91,0x10,0x24]
vfneg.v v2, v1, v0.t 
# ALIAS:    vfabs.v         v2, v1, v0.t     # encoding: [0x57,0x91,0x10,0x28]
# NO-ALIAS: vfsgnjx.vv      v2, v1, v1, v0.t # encoding: [0x57,0x91,0x10,0x28]
vfabs.v v2, v1, v0.t
# ALIAS:    vlm.v           v8, (a0)         # encoding: [0x07,0x04,0xb5,0x02]
# NO-ALIAS: vlm.v           v8, (a0)         # encoding: [0x07,0x04,0xb5,0x02]
vle1.v v8, (a0)
# ALIAS:    vsm.v           v8, (a0)         # encoding: [0x27,0x04,0xb5,0x02]
# NO-ALIAS: vsm.v           v8, (a0)         # encoding: [0x27,0x04,0xb5,0x02]
vse1.v v8, (a0)
# ALIAS:    vfredusum.vs v8, v4, v20, v0.t    # encoding: [0x57,0x14,0x4a,0x04]
# NO-ALIAS: vfredusum.vs v8, v4, v20, v0.t   # encoding: [0x57,0x14,0x4a,0x04]
vfredsum.vs v8, v4, v20, v0.t
# ALIAS:    vfwredusum.vs v8, v4, v20, v0.t   # encoding: [0x57,0x14,0x4a,0xc4]
# NO-ALIAS: vfwredusum.vs v8, v4, v20, v0.t  # encoding: [0x57,0x14,0x4a,0xc4]
vfwredsum.vs v8, v4, v20, v0.t
