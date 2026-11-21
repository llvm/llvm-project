// RUN: not llvm-mc -triple=amdgpu13.10 -filetype=null %s 2>&1 | FileCheck --check-prefix=GFX13-ERR --implicit-check-not=error: --strict-whitespace %s

image_bvh_intersect_ray v[4:7], [v9, v10, v[11:13], v[14:16], v[17:19]], s[4:7]
// GFX13-ERR: :[[@LINE-1]]:1: error: instruction not supported on this GPU

image_bvh_intersect_ray v[4:7], [v9, v10, v[11:13], v[14:16]], s[4:7] a16
// GFX13-ERR: :[[@LINE-1]]:1: error: instruction not supported on this GPU

image_bvh64_intersect_ray v[4:7], [v[9:10], v11, v[12:14], v[15:17], v[18:20]], s[4:7]
// GFX13-ERR: :[[@LINE-1]]:1: error: instruction not supported on this GPU

image_bvh64_intersect_ray v[4:7], [v[9:10], v11, v[12:14], v[15:17]], s[4:7] a16
// GFX13-ERR: :[[@LINE-1]]:1: error: instruction not supported on this GPU
