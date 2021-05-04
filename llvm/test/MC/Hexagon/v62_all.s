# RUN: llvm-mc -arch=hexagon -mcpu=hexagonv62 -filetype=obj -mhvx %s | llvm-objdump --arch=hexagon --mcpu=hexagonv62 --mattr=+hvx -d - | FileCheck %s

//   V6_lvsplatb
//   Vd32.b=vsplat(Rt32)
     V0.b=vsplat(R0)
# CHECK: 19c0c040 { v0.b = vsplat(r0) }

//   V6_lvsplath
//   Vd32.h=vsplat(Rt32)
     V0.h=vsplat(R0)
# CHECK: 19c0c020 { v0.h = vsplat(r0) }

//   V6_pred_scalar2v2
//   Qd4=vsetq2(Rt32)
     Q0=vsetq2(R0)
# CHECK: 19a0c04c { q0 = vsetq2(r0) }

//   V6_shuffeqh
//   Qd4.b=vshuffe(Qs4.h,Qt4.h)
     Q0.b=vshuffe(Q0.h,Q0.h)
# CHECK: 1e03c018 { q0.b = vshuffe(q0.h,q0.h) }

//   V6_shuffeqw
//   Qd4.h=vshuffe(Qs4.w,Qt4.w)
     Q0.h=vshuffe(Q0.w,Q0.w)
# CHECK: 1e03c01c { q0.h = vshuffe(q0.w,q0.w) }

//   V6_vaddbsat
//   Vd32.b=vadd(Vu32.b,Vv32.b):sat
     V0.b=vadd(V0.b,V0.b):sat
# CHECK: 1f00c000 { v0.b = vadd(v0.b,v0.b):sat }

//   V6_vaddbsat_dv
//   Vdd32.b=vadd(Vuu32.b,Vvv32.b):sat
     V1:0.b=vadd(V1:0.b,V1:0.b):sat
# CHECK: 1ea0c000 { v1:0.b = vadd(v1:0.b,v1:0.b):sat }

//   V6_vaddcarry
//   Vd32.w=vadd(Vu32.w,Vv32.w,Qx4):carry
     V0.w=vadd(V0.w,V0.w,Q0):carry
# CHECK: 1ca0e000 { v0.w = vadd(v0.w,v0.w,q0):carry }

//   V6_vaddclbh
//   $Vd.h=vadd(vclb($Vu.h),$Vv.h)
     V0.h=vadd(vclb(V0.h),V0.h)
# CHECK: 1f00e000 { v0.h = vadd(vclb(v0.h),v0.h) }

//   V6_vaddclbw
//   $Vd.w=vadd(vclb($Vu.w),$Vv.w)
     V0.w=vadd(vclb(V0.w),V0.w)
# CHECK: 1f00e020 { v0.w = vadd(vclb(v0.w),v0.w) }

//   V6_vaddhw_acc
//   Vxx32.w+=vadd(Vu32.h,Vv32.h)
     V1:0.w+=vadd(V0.h,V0.h)
# CHECK: 1c20e040 { v1:0.w += vadd(v0.h,v0.h) }

//   V6_vaddubh_acc
//   Vxx32.h+=vadd(Vu32.ub,Vv32.ub)
     V1:0.h+=vadd(V0.ub,V0.ub)
# CHECK: 1c40e0a0 { v1:0.h += vadd(v0.ub,v0.ub) }

//   V6_vaddububb_sat
//   Vd32.ub=vadd(Vu32.ub,Vv32.b):sat
     V0.ub=vadd(V0.ub,V0.b):sat
# CHECK: 1ea0c080 { v0.ub = vadd(v0.ub,v0.b):sat }

//   V6_vadduhw_acc
//   Vxx32.w+=vadd(Vu32.uh,Vv32.uh)
     V1:0.w+=vadd(V0.uh,V0.uh)
# CHECK: 1c40e080 { v1:0.w += vadd(v0.uh,v0.uh) }

//   V6_vadduwsat
//   Vd32.uw=vadd(Vu32.uw,Vv32.uw):sat
     V0.uw=vadd(V0.uw,V0.uw):sat
# CHECK: 1f60c020 { v0.uw = vadd(v0.uw,v0.uw):sat }

//   V6_vadduwsat_dv
//   Vdd32.uw=vadd(Vuu32.uw,Vvv32.uw):sat
     V1:0.uw=vadd(V1:0.uw,V1:0.uw):sat
# CHECK: 1ea0c040 { v1:0.uw = vadd(v1:0.uw,v1:0.uw):sat }

//   V6_vandnqrt
//   Vd32=vand(!Qu4,Rt32)
     V0=vand(!Q0,R0)
# CHECK: 19a0c4a0 { v0 = vand(!q0,r0) }

//   V6_vandnqrt_acc
//   Vx32|=vand(!Qu4,Rt32)
     V0|=vand(!Q0,R0)
# CHECK: 1960e460 { v0 |= vand(!q0,r0) }

//   V6_vandvnqv
//   Vd32=vand(!Qv4,Vu32)
     V0=vand(!Q0,V0)
# CHECK: 1e03e020 { v0 = vand(!q0,v0) }

//   V6_vandvqv
//   Vd32=vand(Qv4,Vu32)
     V0=vand(Q0,V0)
# CHECK: 1e03e000 { v0 = vand(q0,v0) }

//   V6_vasrhbsat
//   Vd32.b=vasr(Vu32.h,Vv32.h,Rt8):sat
     V0.b=vasr(V0.h,V0.h,R0):sat
# CHECK: 1800c000 { v0.b = vasr(v0.h,v0.h,r0):sat }

//   V6_vasruwuhrndsat
//   Vd32.uh=vasr(Vu32.uw,Vv32.uw,Rt8):rnd:sat
     V0.uh=vasr(V0.uw,V0.uw,R0):rnd:sat
# CHECK: 1800c020 { v0.uh = vasr(v0.uw,v0.uw,r0):rnd:sat }

//   V6_vasrwuhrndsat
//   Vd32.uh=vasr(Vu32.w,Vv32.w,Rt8):rnd:sat
     V0.uh=vasr(V0.w,V0.w,R0):rnd:sat
# CHECK: 1800c040 { v0.uh = vasr(v0.w,v0.w,r0):rnd:sat }

//   V6_vL32b_cur_npred_ai
//   if (!Pv4) Vd32.cur=vmem(Rt32+#s4)
     {
     v1=v0
     if (!P0) V0.cur=vmem(R0+#04)
     }
# CHECK: 1e0360e1 { v1 = v0
# CHECK: 2880c4a0   if (!p0) v0.cur = vmem(r0+#4) }

//   V6_vL32b_cur_npred_pi
//   if (!Pv4) Vd32.cur=vmem(Rx32++#s3)
     {
     v1=v0
     if (!P0) V0.cur=vmem(R0++#03)
     }
# CHECK: 1e0360e1 { v1 = v0
# CHECK: 2980c3a0   if (!p0) v0.cur = vmem(r0++#3) }

//   V6_vL32b_cur_npred_ppu
//   if (!Pv4) Vd32.cur=vmem(Rx32++Mu2)
     {
     v1=v0
     if (!P0) V0.cur=vmem(R0++M0)
     }
# CHECK: 1e0360e1 { v1 = v0
# CHECK: 2b80c0a0   if (!p0) v0.cur = vmem(r0++m0) }

//   V6_vL32b_cur_pred_ai
//   if (Pv4) Vd32.cur=vmem(Rt32+#s4)
     {
     v1=v0
     if (P0) V0.cur=vmem(R0+#04)
     }
# CHECK: 1e0360e1 { v1 = v0
# CHECK: 2880c480   if (p0) v0.cur = vmem(r0+#4) }

//   V6_vL32b_cur_pred_pi
//   if (Pv4) Vd32.cur=vmem(Rx32++#s3)
     {
     v1=v0
     if (P0) V0.cur=vmem(R0++#03)
     }
# CHECK: 1e0360e1 { v1 = v0
# CHECK: 2980c380   if (p0) v0.cur = vmem(r0++#3) }

//   V6_vL32b_cur_pred_ppu
//   if (Pv4) Vd32.cur=vmem(Rx32++Mu2)
     {
     v1=v0
     if (P0) V0.cur=vmem(R0++M0)
     }
# CHECK: 1e0360e1 { v1 = v0
# CHECK: 2b80c080   if (p0) v0.cur = vmem(r0++m0) }

//   V6_vL32b_npred_ai
//   if (!Pv4) Vd32=vmem(Rt32+#s4)
     if (!P0) V0=vmem(R0+#04)
# CHECK: 2880c460 { if (!p0) v0 = vmem(r0+#4) }

//   V6_vL32b_npred_pi
//   if (!Pv4) Vd32=vmem(Rx32++#s3)
     if (!P0) V0=vmem(R0++#03)
# CHECK: 2980c360 { if (!p0) v0 = vmem(r0++#3) }

//   V6_vL32b_npred_ppu
//   if (!Pv4) Vd32=vmem(Rx32++Mu2)
     if (!P0) V0=vmem(R0++M0)
# CHECK: 2b80c060 { if (!p0) v0 = vmem(r0++m0) }

//   V6_vL32b_nt_cur_npred_ai
//   if (!Pv4) Vd32.cur=vmem(Rt32+#s4):nt
     {
     v1=v0
     if (!P0) V0.cur=vmem(R0+#04):nt
     }
# CHECK: 1e0360e1 { v1 = v0
# CHECK: 28c0c4a0   if (!p0) v0.cur = vmem(r0+#4):nt }

//   V6_vL32b_nt_cur_npred_pi
//   if (!Pv4) Vd32.cur=vmem(Rx32++#s3):nt
     {
     v1=v0
     if (!P0) V0.cur=vmem(R0++#03):nt
     }
# CHECK: 1e0360e1 { v1 = v0
# CHECK: 29c0c3a0   if (!p0) v0.cur = vmem(r0++#3):nt }

//   V6_vL32b_nt_cur_npred_ppu
//   if (!Pv4) Vd32.cur=vmem(Rx32++Mu2):nt
     {
     v1=v0
     if (!P0) V0.cur=vmem(R0++M0):nt
     }
# CHECK: 1e0360e1 { v1 = v0
# CHECK: 2bc0c0a0   if (!p0) v0.cur = vmem(r0++m0):nt }

//   V6_vL32b_nt_cur_pred_ai
//   if (Pv4) Vd32.cur=vmem(Rt32+#s4):nt
     {
     v1=v0
     if (P0) V0.cur=vmem(R0+#04):nt
     }
# CHECK: 1e0360e1 { v1 = v0
# CHECK: 28c0c480   if (p0) v0.cur = vmem(r0+#4):nt }

//   V6_vL32b_nt_cur_pred_pi
//   if (Pv4) Vd32.cur=vmem(Rx32++#s3):nt
     {
     v1=v0
     if (P0) V0.cur=vmem(R0++#03):nt
     }
# CHECK: 1e0360e1 { v1 = v0
# CHECK: 29c0c380   if (p0) v0.cur = vmem(r0++#3):nt }

//   V6_vL32b_nt_cur_pred_ppu
//   if (Pv4) Vd32.cur=vmem(Rx32++Mu2):nt
     {
     v1=v0
     if (P0) V0.cur=vmem(R0++M0):nt
     }
# CHECK: 1e0360e1 { v1 = v0
# CHECK: 2bc0c080   if (p0) v0.cur = vmem(r0++m0):nt }

//   V6_vL32b_nt_npred_ai
//   if (!Pv4) Vd32=vmem(Rt32+#s4):nt
     if (!P0) V0=vmem(R0+#04):nt
# CHECK: 28c0c460 { if (!p0) v0 = vmem(r0+#4):nt }

//   V6_vL32b_nt_npred_pi
//   if (!Pv4) Vd32=vmem(Rx32++#s3):nt
     if (!P0) V0=vmem(R0++#03):nt
# CHECK: 29c0c360 { if (!p0) v0 = vmem(r0++#3):nt }

//   V6_vL32b_nt_npred_ppu
//   if (!Pv4) Vd32=vmem(Rx32++Mu2):nt
     if (!P0) V0=vmem(R0++M0):nt
# CHECK: 2bc0c060 { if (!p0) v0 = vmem(r0++m0):nt }

//   V6_vL32b_nt_pred_ai
//   if (Pv4) Vd32=vmem(Rt32+#s4):nt
     if (P0) V0=vmem(R0+#04):nt
# CHECK: 28c0c440 { if (p0) v0 = vmem(r0+#4):nt }

//   V6_vL32b_nt_pred_pi
//   if (Pv4) Vd32=vmem(Rx32++#s3):nt
     if (P0) V0=vmem(R0++#03):nt
# CHECK: 29c0c340 { if (p0) v0 = vmem(r0++#3):nt }

//   V6_vL32b_nt_pred_ppu
//   if (Pv4) Vd32=vmem(Rx32++Mu2):nt
     if (P0) V0=vmem(R0++M0):nt
# CHECK: 2bc0c040 { if (p0) v0 = vmem(r0++m0):nt }

//   V6_vL32b_nt_tmp_npred_ai
//   if (!Pv4) Vd32.tmp=vmem(Rt32+#s4):nt
     {
     v1=v0
     if (!P0) V0.tmp=vmem(R0+#04):nt
     }
# CHECK: 1e0360e1 { v1 = v0
# CHECK: 28c0c4e0   if (!p0) v0.tmp = vmem(r0+#4):nt }

//   V6_vL32b_nt_tmp_npred_pi
//   if (!Pv4) Vd32.tmp=vmem(Rx32++#s3):nt
     {
     v1=v0
     if (!P0) V0.tmp=vmem(R0++#03):nt
     }
# CHECK: 1e0360e1 { v1 = v0
# CHECK: 29c0c3e0   if (!p0) v0.tmp = vmem(r0++#3):nt }

//   V6_vL32b_nt_tmp_npred_ppu
//   if (!Pv4) Vd32.tmp=vmem(Rx32++Mu2):nt
     {
     v1=v0
     if (!P0) V0.tmp=vmem(R0++M0):nt
     }
# CHECK: 1e0360e1 { v1 = v0
# CHECK: 2bc0c0e0   if (!p0) v0.tmp = vmem(r0++m0):nt }

//   V6_vL32b_nt_tmp_pred_ai
//   if (Pv4) Vd32.tmp=vmem(Rt32+#s4):nt
     {
     v1=v0
     if (P0) V0.tmp=vmem(R0+#04):nt
     }
# CHECK: 1e0360e1 { v1 = v0
# CHECK: 28c0c4c0   if (p0) v0.tmp = vmem(r0+#4):nt }

//   V6_vL32b_nt_tmp_pred_pi
//   if (Pv4) Vd32.tmp=vmem(Rx32++#s3):nt
     {
     v1=v0
     if (P0) V0.tmp=vmem(R0++#03):nt
     }
# CHECK: 1e0360e1 { v1 = v0
# CHECK: 29c0c3c0   if (p0) v0.tmp = vmem(r0++#3):nt }

//   V6_vL32b_nt_tmp_pred_ppu
//   if (Pv4) Vd32.tmp=vmem(Rx32++Mu2):nt
     {
     v1=v0
     if (P0) V0.tmp=vmem(R0++M0):nt
     }
# CHECK: 1e0360e1 { v1 = v0
# CHECK: 2bc0c0c0   if (p0) v0.tmp = vmem(r0++m0):nt }

//   V6_vL32b_pred_ai
//   if (Pv4) Vd32=vmem(Rt32+#s4)
     if (P0) V0=vmem(R0+#04)
# CHECK: 2880c440 { if (p0) v0 = vmem(r0+#4) }

//   V6_vL32b_pred_pi
//   if (Pv4) Vd32=vmem(Rx32++#s3)
     if (P0) V0=vmem(R0++#03)
# CHECK: 2980c340 { if (p0) v0 = vmem(r0++#3) }

//   V6_vL32b_pred_ppu
//   if (Pv4) Vd32=vmem(Rx32++Mu2)
     if (P0) V0=vmem(R0++M0)
# CHECK: 2b80c040 { if (p0) v0 = vmem(r0++m0) }

//   V6_vL32b_tmp_npred_ai
//   if (!Pv4) Vd32.tmp=vmem(Rt32+#s4)
     {
     v1=v0
     if (!P0) V0.tmp=vmem(R0+#04)
     }
# CHECK: 1e0360e1 { v1 = v0
# CHECK: 2880c4e0   if (!p0) v0.tmp = vmem(r0+#4) }

//   V6_vL32b_tmp_npred_pi
//   if (!Pv4) Vd32.tmp=vmem(Rx32++#s3)
     {
     v1=v0
     if (!P0) V0.tmp=vmem(R0++#03)
     }
# CHECK: 1e0360e1 { v1 = v0
# CHECK: 2980c3e0   if (!p0) v0.tmp = vmem(r0++#3) }

//   V6_vL32b_tmp_npred_ppu
//   if (!Pv4) Vd32.tmp=vmem(Rx32++Mu2)
     {
     v1=v0
     if (!P0) V0.tmp=vmem(R0++M0)
     }
# CHECK: 1e0360e1 { v1 = v0
# CHECK: 2b80c0e0   if (!p0) v0.tmp = vmem(r0++m0) }

//   V6_vL32b_tmp_pred_ai
//   if (Pv4) Vd32.tmp=vmem(Rt32+#s4)
     {
     v1=v0
     if (P0) V0.tmp=vmem(R0+#04)
     }
# CHECK: 1e0360e1 { v1 = v0
# CHECK: 2880c4c0   if (p0) v0.tmp = vmem(r0+#4) }

//   V6_vL32b_tmp_pred_pi
//   if (Pv4) Vd32.tmp=vmem(Rx32++#s3)
     {
     v1=v0
     if (P0) V0.tmp=vmem(R0++#03)
     }
# CHECK: 1e0360e1 { v1 = v0
# CHECK: 2980c3c0   if (p0) v0.tmp = vmem(r0++#3) }

//   V6_vL32b_tmp_pred_ppu
//   if (Pv4) Vd32.tmp=vmem(Rx32++Mu2)
     {
     v1=v0
     if (P0) V0.tmp=vmem(R0++M0)
     }
# CHECK: 1e0360e1 { v1 = v0
# CHECK: 2b80c0c0   if (p0) v0.tmp = vmem(r0++m0) }

//   V6_vlsrb
//   Vd32.ub=vlsr(Vu32.ub,Rt32)
     V0.ub=vlsr(V0.ub,R0)
# CHECK: 1980c060 { v0.ub = vlsr(v0.ub,r0) }

//   V6_vlutvvbi
//   Vd32.b=vlut32(Vu32.b,Vv32.b,#u3)
     V0.b=vlut32(V0.b,V0.b,#03)
# CHECK: 1e20c060 { v0.b = vlut32(v0.b,v0.b,#3) }

//   V6_vlutvvb_nm
//   Vd32.b=vlut32(Vu32.b,Vv32.b,Rt8):nomatch
     V0.b=vlut32(V0.b,V0.b,R0):nomatch
# CHECK: 1800c060 { v0.b = vlut32(v0.b,v0.b,r0):nomatch }

//   V6_vlutvvb_oracci
//   Vx32.b|=vlut32(Vu32.b,Vv32.b,#u3)
     V0.b|=vlut32(V0.b,V0.b,#03)
# CHECK: 1cc0e060 { v0.b |= vlut32(v0.b,v0.b,#3) }

//   V6_vlutvwhi
//   Vdd32.h=vlut16(Vu32.b,Vv32.h,#u3)
     V1:0.h=vlut16(V0.b,V0.h,#03)
# CHECK: 1e60c060 { v1:0.h = vlut16(v0.b,v0.h,#3) }

//   V6_vlutvwh_nm
//   Vdd32.h=vlut16(Vu32.b,Vv32.h,Rt8):nomatch
     V1:0.h=vlut16(V0.b,V0.h,R0):nomatch
# CHECK: 1800c080 { v1:0.h = vlut16(v0.b,v0.h,r0):nomatch }

//   V6_vlutvwh_oracci
//   Vxx32.h|=vlut16(Vu32.b,Vv32.h,#u3)
     V1:0.h|=vlut16(V0.b,V0.h,#03)
# CHECK: 1ce0e060 { v1:0.h |= vlut16(v0.b,v0.h,#3) }

//   V6_vmaxb
//   Vd32.b=vmax(Vu32.b,Vv32.b)
     V0.b=vmax(V0.b,V0.b)
# CHECK: 1f20c0a0 { v0.b = vmax(v0.b,v0.b) }

//   V6_vminb
//   Vd32.b=vmin(Vu32.b,Vv32.b)
     V0.b=vmin(V0.b,V0.b)
# CHECK: 1f20c080 { v0.b = vmin(v0.b,v0.b) }

//   V6_vmpauhb
//   Vdd32.w=vmpa(Vuu32.uh,Rt32.b)
     V1:0.w=vmpa(V1:0.uh,R0.b)
# CHECK: 1980c0a0 { v1:0.w = vmpa(v1:0.uh,r0.b) }

//   V6_vmpauhb_acc
//   Vxx32.w+=vmpa(Vuu32.uh,Rt32.b)
     V1:0.w+=vmpa(V1:0.uh,R0.b)
# CHECK: 1980e040 { v1:0.w += vmpa(v1:0.uh,r0.b) }

//   V6_vmpyewuh_64
//   Vdd32=vmpye(Vu32.w,Vv32.uh)
     V1:0=vmpye(V0.w,V0.uh)
# CHECK: 1ea0c0c0 { v1:0 = vmpye(v0.w,v0.uh) }

//   V6_vmpyiwub
//   Vd32.w=vmpyi(Vu32.w,Rt32.ub)
     V0.w=vmpyi(V0.w,R0.ub)
# CHECK: 1980c0c0 { v0.w = vmpyi(v0.w,r0.ub) }

//   V6_vmpyiwub_acc
//   Vx32.w+=vmpyi(Vu32.w,Rt32.ub)
     V0.w+=vmpyi(V0.w,R0.ub)
# CHECK: 1980e020 { v0.w += vmpyi(v0.w,r0.ub) }

//   V6_vmpyowh_64_acc
//   Vxx32+=vmpyo(Vu32.w,Vv32.h)
     V1:0+=vmpyo(V0.w,V0.h)
# CHECK: 1c20e060 { v1:0 += vmpyo(v0.w,v0.h) }

//   V6_vrounduhub
//   Vd32.ub=vround(Vu32.uh,Vv32.uh):sat
     V0.ub=vround(V0.uh,V0.uh):sat
# CHECK: 1fe0c060 { v0.ub = vround(v0.uh,v0.uh):sat }

//   V6_vrounduwuh
//   Vd32.uh=vround(Vu32.uw,Vv32.uw):sat
     V0.uh=vround(V0.uw,V0.uw):sat
# CHECK: 1fe0c080 { v0.uh = vround(v0.uw,v0.uw):sat }

//   V6_vsatuwuh
//   Vd32.uh=vsat(Vu32.uw,Vv32.uw)
     V0.uh=vsat(V0.uw,V0.uw)
# CHECK: 1f20c0c0 { v0.uh = vsat(v0.uw,v0.uw) }

//   V6_vsubbsat
//   Vd32.b=vsub(Vu32.b,Vv32.b):sat
     V0.b=vsub(V0.b,V0.b):sat
# CHECK: 1f20c040 { v0.b = vsub(v0.b,v0.b):sat }

//   V6_vsubbsat_dv
//   Vdd32.b=vsub(Vuu32.b,Vvv32.b):sat
     V1:0.b=vsub(V1:0.b,V1:0.b):sat
# CHECK: 1ea0c020 { v1:0.b = vsub(v1:0.b,v1:0.b):sat }

//   V6_vsubcarry
//   Vd32.w=vsub(Vu32.w,Vv32.w,Qx4):carry
     V0.w=vsub(V0.w,V0.w,Q0):carry
# CHECK: 1ca0e080 { v0.w = vsub(v0.w,v0.w,q0):carry }

//   V6_vsubububb_sat
//   Vd32.ub=vsub(Vu32.ub,Vv32.b):sat
     V0.ub=vsub(V0.ub,V0.b):sat
# CHECK: 1ea0c0a0 { v0.ub = vsub(v0.ub,v0.b):sat }

//   V6_vsubuwsat
//   Vd32.uw=vsub(Vu32.uw,Vv32.uw):sat
     V0.uw=vsub(V0.uw,V0.uw):sat
# CHECK: 1fc0c080 { v0.uw = vsub(v0.uw,v0.uw):sat }

//   V6_vsubuwsat_dv
//   Vdd32.uw=vsub(Vuu32.uw,Vvv32.uw):sat
     V1:0.uw=vsub(V1:0.uw,V1:0.uw):sat
# CHECK: 1ea0c060 { v1:0.uw = vsub(v1:0.uw,v1:0.uw):sat }

//   V6_vwhist128
//   vwhist128
     vwhist128
# CHECK: 1e00e480 { vwhist128 }

//   V6_vwhist128m
//   vwhist128(#u1)
     vwhist128(#01)
# CHECK: 1e00e780 { vwhist128(#1) }

//   V6_vwhist128q
//   vwhist128(Qv4)
     vwhist128(Q0)
# CHECK: 1e02e480 { vwhist128(q0) }

//   V6_vwhist128qm
//   vwhist128(Qv4,#u1)
     vwhist128(Q0,#01)
# CHECK: 1e02e780 { vwhist128(q0,#1) }

//   V6_vwhist256
//   vwhist256
     vwhist256
# CHECK: 1e00e280 { vwhist256 }

//   V6_vwhist256q
//   vwhist256(Qv4)
     vwhist256(Q0)
# CHECK: 1e02e280 { vwhist256(q0) }

//   V6_vwhist256q_sat
//   vwhist256(Qv4):sat
     vwhist256(Q0):sat
# CHECK: 1e02e380 { vwhist256(q0):sat }

//   V6_vwhist256_sat
//   vwhist256:sat
     vwhist256:sat
# CHECK: 1e00e380 { vwhist256:sat }
