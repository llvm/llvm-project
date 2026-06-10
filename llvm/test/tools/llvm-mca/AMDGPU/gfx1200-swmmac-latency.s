# RUN: llvm-mca -mtriple=amdgcn -mcpu=gfx1200 --timeline --timeline-max-iterations=1 --iterations=1 < %s | FileCheck %s

.text
test_swmmac_schedule:
  v_swmmac_f32_16x16x32_f16 v[0:7], v[8:9], v[10:13], v[20:27]
  v_add_f32 v14, v14, v14
  v_add_f32 v15, v15, v15
  v_add_f32 v16, v16, v16
  v_add_f32 v17, v17, v17
  v_swmmac_f32_16x16x32_f16 v[0:7], v[8:9], v[10:13], v[20:27]

# CHECK:      v_swmmac_f32_16x16x32_f16
# CHECK:      v_add_f32
# CHECK:      v_add_f32
# CHECK:      v_add_f32
# CHECK:      v_add_f32
# CHECK:      v_swmmac_f32_16x16x32_f16
