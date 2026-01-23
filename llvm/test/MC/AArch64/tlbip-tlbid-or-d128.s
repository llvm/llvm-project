// NOTE: These TLBIP forms are valid with either +tlbid or +d128.
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+tlbid,+tlb-rmi,+xs < %s | FileCheck %s --check-prefix=TLBID
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+d128,+tlb-rmi,+xs < %s | FileCheck %s --check-prefix=D128

tlbip VAE1OS, x0, x1
// TLBID: tlbip vae1os, x0, x1
// D128: tlbip vae1os, x0, x1

tlbip VAAE1OS, x0, x1
// TLBID: tlbip vaae1os, x0, x1
// D128: tlbip vaae1os, x0, x1

tlbip VALE1OS, x0, x1
// TLBID: tlbip vale1os, x0, x1
// D128: tlbip vale1os, x0, x1

tlbip VAALE1OS, x0, x1
// TLBID: tlbip vaale1os, x0, x1
// D128: tlbip vaale1os, x0, x1

tlbip RVAE1IS, x0, x1
// TLBID: tlbip rvae1is, x0, x1
// D128: tlbip rvae1is, x0, x1

tlbip RVAAE1IS, x0, x1
// TLBID: tlbip rvaae1is, x0, x1
// D128: tlbip rvaae1is, x0, x1

tlbip RVALE1IS, x0, x1
// TLBID: tlbip rvale1is, x0, x1
// D128: tlbip rvale1is, x0, x1

tlbip RVAALE1IS, x0, x1
// TLBID: tlbip rvaale1is, x0, x1
// D128: tlbip rvaale1is, x0, x1

tlbip VAE1IS, x0, x1
// TLBID: tlbip vae1is, x0, x1
// D128: tlbip vae1is, x0, x1

tlbip VAAE1IS, x0, x1
// TLBID: tlbip vaae1is, x0, x1
// D128: tlbip vaae1is, x0, x1

tlbip VALE1IS, x0, x1
// TLBID: tlbip vale1is, x0, x1
// D128: tlbip vale1is, x0, x1

tlbip VAALE1IS, x0, x1
// TLBID: tlbip vaale1is, x0, x1
// D128: tlbip vaale1is, x0, x1

tlbip RVAE1OS, x0, x1
// TLBID: tlbip rvae1os, x0, x1
// D128: tlbip rvae1os, x0, x1

tlbip RVAAE1OS, x0, x1
// TLBID: tlbip rvaae1os, x0, x1
// D128: tlbip rvaae1os, x0, x1

tlbip RVALE1OS, x0, x1
// TLBID: tlbip rvale1os, x0, x1
// D128: tlbip rvale1os, x0, x1

tlbip RVAALE1OS, x0, x1
// TLBID: tlbip rvaale1os, x0, x1
// D128: tlbip rvaale1os, x0, x1

tlbip VAE1OSNXS, x0, x1
// TLBID: tlbip vae1osnxs, x0, x1
// D128: tlbip vae1osnxs, x0, x1

tlbip VAAE1OSNXS, x0, x1
// TLBID: tlbip vaae1osnxs, x0, x1
// D128: tlbip vaae1osnxs, x0, x1

tlbip VALE1OSNXS, x0, x1
// TLBID: tlbip vale1osnxs, x0, x1
// D128: tlbip vale1osnxs, x0, x1

tlbip VAALE1OSNXS, x0, x1
// TLBID: tlbip vaale1osnxs, x0, x1
// D128: tlbip vaale1osnxs, x0, x1

tlbip RVAE1ISNXS, x0, x1
// TLBID: tlbip rvae1isnxs, x0, x1
// D128: tlbip rvae1isnxs, x0, x1

tlbip RVAAE1ISNXS, x0, x1
// TLBID: tlbip rvaae1isnxs, x0, x1
// D128: tlbip rvaae1isnxs, x0, x1

tlbip RVALE1ISNXS, x0, x1
// TLBID: tlbip rvale1isnxs, x0, x1
// D128: tlbip rvale1isnxs, x0, x1

tlbip RVAALE1ISNXS, x0, x1
// TLBID: tlbip rvaale1isnxs, x0, x1
// D128: tlbip rvaale1isnxs, x0, x1

tlbip VAE1ISNXS, x0, x1
// TLBID: tlbip vae1isnxs, x0, x1
// D128: tlbip vae1isnxs, x0, x1

tlbip VAAE1ISNXS, x0, x1
// TLBID: tlbip vaae1isnxs, x0, x1
// D128: tlbip vaae1isnxs, x0, x1

tlbip VALE1ISNXS, x0, x1
// TLBID: tlbip vale1isnxs, x0, x1
// D128: tlbip vale1isnxs, x0, x1

tlbip VAALE1ISNXS, x0, x1
// TLBID: tlbip vaale1isnxs, x0, x1
// D128: tlbip vaale1isnxs, x0, x1

tlbip RVAE1OSNXS, x0, x1
// TLBID: tlbip rvae1osnxs, x0, x1
// D128: tlbip rvae1osnxs, x0, x1

tlbip RVAAE1OSNXS, x0, x1
// TLBID: tlbip rvaae1osnxs, x0, x1
// D128: tlbip rvaae1osnxs, x0, x1

tlbip RVALE1OSNXS, x0, x1
// TLBID: tlbip rvale1osnxs, x0, x1
// D128: tlbip rvale1osnxs, x0, x1

tlbip RVAALE1OSNXS, x0, x1
// TLBID: tlbip rvaale1osnxs, x0, x1
// D128: tlbip rvaale1osnxs, x0, x1

tlbip IPAS2E1IS, x0, x1
// TLBID: tlbip ipas2e1is, x0, x1
// D128: tlbip ipas2e1is, x0, x1

tlbip RIPAS2E1IS, x0, x1
// TLBID: tlbip ripas2e1is, x0, x1
// D128: tlbip ripas2e1is, x0, x1

tlbip IPAS2LE1IS, x0, x1
// TLBID: tlbip ipas2le1is, x0, x1
// D128: tlbip ipas2le1is, x0, x1

tlbip RIPAS2LE1IS, x0, x1
// TLBID: tlbip ripas2le1is, x0, x1
// D128: tlbip ripas2le1is, x0, x1

tlbip VAE2OS, x0, x1
// TLBID: tlbip vae2os, x0, x1
// D128: tlbip vae2os, x0, x1

tlbip VALE2OS, x0, x1
// TLBID: tlbip vale2os, x0, x1
// D128: tlbip vale2os, x0, x1

tlbip RVAE2IS, x0, x1
// TLBID: tlbip rvae2is, x0, x1
// D128: tlbip rvae2is, x0, x1

tlbip RVALE2IS, x0, x1
// TLBID: tlbip rvale2is, x0, x1
// D128: tlbip rvale2is, x0, x1

tlbip VAE2IS, x0, x1
// TLBID: tlbip vae2is, x0, x1
// D128: tlbip vae2is, x0, x1

tlbip VALE2IS, x0, x1
// TLBID: tlbip vale2is, x0, x1
// D128: tlbip vale2is, x0, x1

tlbip IPAS2E1OS, x0, x1
// TLBID: tlbip ipas2e1os, x0, x1
// D128: tlbip ipas2e1os, x0, x1

tlbip RIPAS2E1OS, x0, x1
// TLBID: tlbip ripas2e1os, x0, x1
// D128: tlbip ripas2e1os, x0, x1

tlbip IPAS2LE1OS, x0, x1
// TLBID: tlbip ipas2le1os, x0, x1
// D128: tlbip ipas2le1os, x0, x1

tlbip RIPAS2LE1OS, x0, x1
// TLBID: tlbip ripas2le1os, x0, x1
// D128: tlbip ripas2le1os, x0, x1

tlbip RVAE2OS, x0, x1
// TLBID: tlbip rvae2os, x0, x1
// D128: tlbip rvae2os, x0, x1

tlbip RVALE2OS, x0, x1
// TLBID: tlbip rvale2os, x0, x1
// D128: tlbip rvale2os, x0, x1

tlbip IPAS2E1ISNXS, x0, x1
// TLBID: tlbip ipas2e1isnxs, x0, x1
// D128: tlbip ipas2e1isnxs, x0, x1

tlbip RIPAS2E1ISNXS, x0, x1
// TLBID: tlbip ripas2e1isnxs, x0, x1
// D128: tlbip ripas2e1isnxs, x0, x1

tlbip IPAS2LE1ISNXS, x0, x1
// TLBID: tlbip ipas2le1isnxs, x0, x1
// D128: tlbip ipas2le1isnxs, x0, x1

tlbip RIPAS2LE1ISNXS, x0, x1
// TLBID: tlbip ripas2le1isnxs, x0, x1
// D128: tlbip ripas2le1isnxs, x0, x1

tlbip VAE2OSNXS, x0, x1
// TLBID: tlbip vae2osnxs, x0, x1
// D128: tlbip vae2osnxs, x0, x1

tlbip VALE2OSNXS, x0, x1
// TLBID: tlbip vale2osnxs, x0, x1
// D128: tlbip vale2osnxs, x0, x1

tlbip RVAE2ISNXS, x0, x1
// TLBID: tlbip rvae2isnxs, x0, x1
// D128: tlbip rvae2isnxs, x0, x1

tlbip RVALE2ISNXS, x0, x1
// TLBID: tlbip rvale2isnxs, x0, x1
// D128: tlbip rvale2isnxs, x0, x1

tlbip VAE2ISNXS, x0, x1
// TLBID: tlbip vae2isnxs, x0, x1
// D128: tlbip vae2isnxs, x0, x1

tlbip VALE2ISNXS, x0, x1
// TLBID: tlbip vale2isnxs, x0, x1
// D128: tlbip vale2isnxs, x0, x1

tlbip IPAS2E1OSNXS, x0, x1
// TLBID: tlbip ipas2e1osnxs, x0, x1
// D128: tlbip ipas2e1osnxs, x0, x1

tlbip RIPAS2E1OSNXS, x0, x1
// TLBID: tlbip ripas2e1osnxs, x0, x1
// D128: tlbip ripas2e1osnxs, x0, x1

tlbip IPAS2LE1OSNXS, x0, x1
// TLBID: tlbip ipas2le1osnxs, x0, x1
// D128: tlbip ipas2le1osnxs, x0, x1

tlbip RIPAS2LE1OSNXS, x0, x1
// TLBID: tlbip ripas2le1osnxs, x0, x1
// D128: tlbip ripas2le1osnxs, x0, x1

tlbip RVAE2OSNXS, x0, x1
// TLBID: tlbip rvae2osnxs, x0, x1
// D128: tlbip rvae2osnxs, x0, x1

tlbip RVALE2OSNXS, x0, x1
// TLBID: tlbip rvale2osnxs, x0, x1
// D128: tlbip rvale2osnxs, x0, x1
