# RUN: llvm-mc -filetype=obj -triple=hexagon %s | llvm-objdump -r - | FileCheck %s

# CHECK: R_HEX_B22_PCREL
r_hex_b22_pcrel:
{ jump #undefined }

# CHECK: R_HEX_B15_PCREL
r_hex_b15_pcrel:
{ if (p0) jump #undefined }

# CHECK: R_HEX_B7_PCREL
r_hex_b7_pcrel:
{ loop1 (#undefined, #0) }

# CHECK: R_HEX_32
r_hex_32:
.word undefined

# CHECK: R_HEX_16
r_hex_16:
.half undefined
.half 0

# CHECK: R_HEX_8
r_hex_8:
.byte undefined
.byte 0
.byte 0
.byte 0

# CHECK: R_HEX_GPREL16_0
r_hex_gprel16_0:
{ r0 = memb (#undefined@gotrel) }

# CHECK: R_HEX_GPREL16_1
r_hex_gprel16_1:
{ r0 = memh (#undefined@gotrel) }

# CHECK: R_HEX_GPREL16_2
r_hex_gprel16_2:
{ r0 = memw (#undefined@gotrel) }

# CHECK: R_HEX_GPREL16_3
r_hex_gprel16_3:
{ r1:0 = memd (#undefined@gotrel) }

# CHECK: R_HEX_B13_PCREL
r_hex_b13_pcrel:
{ if (r0 != #0) jump:nt #undefined }

# CHECK: R_HEX_B9_PCREL
r_hex_b9_pcrel:
{ r0 = #0 ; jump #undefined }

# CHECK: R_HEX_B32_PCREL_X
r_hex_b32_pcrel_x:
{ jump ##undefined }

# CHECK: R_HEX_32_6_X
r_hex_32_6_x:
{ r0 = ##undefined }

# CHECK: R_HEX_B22_PCREL_X
r_hex_b22_pcrel_x:
{ jump ##undefined }

# CHECK: R_HEX_B15_PCREL_X
r_hex_b15_pcrel_x:
{ if (p0) jump ##undefined }

# CHECK: R_HEX_B9_PCREL_X
r_hex_b9_pcrel_x:
{ r0 = #0 ; jump ##undefined }

# CHECK: R_HEX_B7_PCREL_X
r_hex_b7_pcrel_x:
{ loop1 (##undefined, #0) }

# CHECK: R_HEX_32_PCREL
r_hex_32_pcrel:
.word undefined@pcrel

# CHECK: R_HEX_PLT_B22_PCREL
r_hex_plt_b22_pcrel:
jump undefined@plt

# CHECK: R_HEX_GOTREL_32
r_hex_gotrel_32:
.word undefined@gotrel

# CHECK: R_HEX_GOT_32
r_hex_got_32:
.word undefined@got

# CHECK: R_HEX_GOT_16
r_hex_got_16:
.half undefined@got
.half 0

# CHECK: R_HEX_DTPREL_32
r_hex_dtprel_32:
.word undefined@dtprel

# CHECK: R_HEX_DTPREL_16
r_hex_dtprel_16:
.half undefined@dtprel
.half 0

# CHECK: R_HEX_GD_GOT_32
r_hex_gd_got_32:
.word undefined@gdgot

# CHECK: R_HEX_GD_GOT_16
r_hex_gd_got_16:
.half undefined@gdgot
.half 0

# CHECK: R_HEX_IE_32
r_hex_ie_32:
.word undefined@ie

# CHECK: R_HEX_IE_GOT_32
r_hex_ie_got_32:
.word undefined@iegot

# CHECK: R_HEX_IE_GOT_16
r_hex_ie_got_16:
.half undefined@iegot
.half 0

# CHECK: R_HEX_TPREL_32
r_hex_tprel_32:
.word undefined@tprel

# CHECK: R_HEX_TPREL_16
r_hex_tprel_16:
r0 = #undefined@tprel

# CHECK: R_HEX_6_PCREL_X
r_hex_6_pcrel_x:
{ r0 = ##undefined@pcrel
  r1 = r1 }

# CHECK: R_HEX_GOTREL_32_6_X
r_hex_gotrel_32_6_x:
{ r0 = ##undefined@gotrel }

# CHECK: R_HEX_GOTREL_16_X
r_hex_gotrel_16_x:
{ r0 = ##undefined@gotrel }

# CHECK: R_HEX_GOTREL_11_X
r_hex_gotrel_11_x:
{ r0 = memw(r0 + ##undefined@gotrel) }

# CHECK: R_HEX_GOT_32_6_X
r_hex_got_32_6_x:
{ r0 = ##undefined@got }

# CHECK: R_HEX_GOT_16_X
r_hex_got_16_x:
{ r0 = ##undefined@got }

# CHECK: R_HEX_GOT_11_X
r_hex_got_11_x:
{ r0 = memw(r0 + ##undefined@got) }

# CHECK: R_HEX_DTPREL_32_6_X
r_hex_dtprel_32_6_x:
{ r0 = ##undefined@dtprel }

# CHECK: R_HEX_DTPREL_16_X
r_hex_dtprel_16_x:
{ r0 = ##undefined@dtprel }

# CHECK: R_HEX_DTPREL_11_X
r_hex_dtprel_11_x:
{ r0 = memw(r0 + ##undefined@dtprel) }

# CHECK: R_HEX_GD_GOT_32_6_X
r_hex_gd_got_32_6_x:
{ r0 = ##undefined@gdgot }

# CHECK: R_HEX_GD_GOT_16_X
r_hex_gd_got_16_x:
{ r0 = ##undefined@gdgot }

# CHECK: R_HEX_GD_GOT_11_X
r_hex_gd_got_11_x:
{ r0 = memw(r0 + ##undefined@gdgot) }

# CHECK: R_HEX_IE_32_6_X
r_hex_ie_32_6_x:
{ r0 = ##undefined@ie }

# CHECK: R_HEX_IE_16_X
r_hex_ie_16_x:
{ r0 = ##undefined@ie }

# CHECK: R_HEX_IE_GOT_32_6_X
r_hex_ie_got_32_6_x:
{ r0 = ##undefined@iegot }

# CHECK: R_HEX_IE_GOT_16_X
r_hex_ie_got_16_x:
{ r0 = ##undefined@iegot }

# CHECK: R_HEX_IE_GOT_11_X
r_hex_ie_got_11_x:
{ r0 = memw(r0 + ##undefined@iegot) }

# CHECK: R_HEX_TPREL_32_6_X
r_hex_tprel_32_6_x:
{ r0 = ##undefined@tprel }

# CHECK: R_HEX_TPREL_16_X
r_hex_tprel_16_x:
{ r0 = ##undefined@tprel }

# CHECK: R_HEX_TPREL_11_X
r_hex_tprel_11_x:
{ r0 = memw(r0 + ##undefined@tprel) }

# CHECK: R_HEX_LD_GOT_32
r_hex_ld_got_32:
.word undefined@ldgot

# CHECK: R_HEX_LD_GOT_16
r_hex_ld_got_16:
.half undefined@ldgot
.half 0

# CHECK: R_HEX_LD_GOT_32_6_X
r_hex_ld_got_32_6_x:
{ r0 = ##undefined@ldgot }

# CHECK: R_HEX_LD_GOT_16_X
r_hex_ld_got_16_x:
{ r0 = ##undefined@ldgot }

# CHECK: R_HEX_LD_GOT_11_X
r_hex_ld_got_11_x:
{ r0 = memw(r0 + ##undefined@ldgot) }

