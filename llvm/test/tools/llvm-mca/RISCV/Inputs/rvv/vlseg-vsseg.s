# Vector segmented load/store.

vsetvli zero, zero, e8, mf8, tu, mu
vlseg2e8.v  v8,(a0)
vsetvli zero, zero, e8, mf4, tu, mu
vlseg2e8.v  v8,(a0)
vsetvli zero, zero, e8, mf2, tu, mu
vlseg2e8.v  v8,(a0)
vsetvli zero, zero, e8, m1, tu, mu
vlseg2e8.v  v8,(a0)
vsetvli zero, zero, e8, m2, tu, mu
vlseg2e8.v  v8,(a0)
vsetvli zero, zero, e8, m4, tu, mu
vlseg2e8.v  v8,(a0)
vsetvli zero, zero, e16, mf4, tu, mu
vlseg2e16.v v8,(a0)
vsetvli zero, zero, e16, mf2, tu, mu
vlseg2e16.v v8,(a0)
vsetvli zero, zero, e16, m1, tu, mu
vlseg2e16.v v8,(a0)
vsetvli zero, zero, e16, m2, tu, mu
vlseg2e16.v v8,(a0)
vsetvli zero, zero, e16, m4, tu, mu
vlseg2e16.v v8,(a0)
vsetvli zero, zero, e32, mf2, tu, mu
vlseg2e32.v v8,(a0)
vsetvli zero, zero, e32, m1, tu, mu
vlseg2e32.v v8,(a0)
vsetvli zero, zero, e32, m2, tu, mu
vlseg2e32.v v8,(a0)
vsetvli zero, zero, e32, m4, tu, mu
vlseg2e32.v v8,(a0)
vsetvli zero, zero, e64, m1, tu, mu
vlseg2e64.v v8,(a0)
vsetvli zero, zero, e64, m2, tu, mu
vlseg2e64.v v8,(a0)
vsetvli zero, zero, e64, m4, tu, mu
vlseg2e64.v v8,(a0)

vsetvli zero, zero, e8, mf8, tu, mu
vlseg3e8.v  v8,(a0)
vsetvli zero, zero, e8, mf4, tu, mu
vlseg3e8.v  v8,(a0)
vsetvli zero, zero, e8, mf2, tu, mu
vlseg3e8.v  v8,(a0)
vsetvli zero, zero, e8, m1, tu, mu
vlseg3e8.v  v8,(a0)
vsetvli zero, zero, e8, m2, tu, mu
vlseg3e8.v  v8,(a0)
vsetvli zero, zero, e16, mf4, tu, mu
vlseg3e16.v v8,(a0)
vsetvli zero, zero, e16, mf2, tu, mu
vlseg3e16.v v8,(a0)
vsetvli zero, zero, e16, m1, tu, mu
vlseg3e16.v v8,(a0)
vsetvli zero, zero, e16, m2, tu, mu
vlseg3e16.v v8,(a0)
vsetvli zero, zero, e32, mf2, tu, mu
vlseg3e32.v v8,(a0)
vsetvli zero, zero, e32, m1, tu, mu
vlseg3e32.v v8,(a0)
vsetvli zero, zero, e32, m2, tu, mu
vlseg3e32.v v8,(a0)
vsetvli zero, zero, e64, m1, tu, mu
vlseg3e64.v v8,(a0)
vsetvli zero, zero, e64, m2, tu, mu
vlseg3e64.v v8,(a0)

vsetvli zero, zero, e8, mf8, tu, mu
vlseg4e8.v  v8,(a0)
vsetvli zero, zero, e8, mf4, tu, mu
vlseg4e8.v  v8,(a0)
vsetvli zero, zero, e8, mf2, tu, mu
vlseg4e8.v  v8,(a0)
vsetvli zero, zero, e8, m1, tu, mu
vlseg4e8.v  v8,(a0)
vsetvli zero, zero, e8, m2, tu, mu
vlseg4e8.v  v8,(a0)
vsetvli zero, zero, e16, mf4, tu, mu
vlseg4e16.v v8,(a0)
vsetvli zero, zero, e16, mf2, tu, mu
vlseg4e16.v v8,(a0)
vsetvli zero, zero, e16, m1, tu, mu
vlseg4e16.v v8,(a0)
vsetvli zero, zero, e16, m2, tu, mu
vlseg4e16.v v8,(a0)
vsetvli zero, zero, e32, mf2, tu, mu
vlseg4e32.v v8,(a0)
vsetvli zero, zero, e32, m1, tu, mu
vlseg4e32.v v8,(a0)
vsetvli zero, zero, e32, m2, tu, mu
vlseg4e32.v v8,(a0)
vsetvli zero, zero, e64, m1, tu, mu
vlseg4e64.v v8,(a0)
vsetvli zero, zero, e64, m2, tu, mu
vlseg4e64.v v8,(a0)

vsetvli zero, zero, e8, mf8, tu, mu
vlseg5e8.v  v8,(a0)
vsetvli zero, zero, e8, mf4, tu, mu
vlseg5e8.v  v8,(a0)
vsetvli zero, zero, e8, mf2, tu, mu
vlseg5e8.v  v8,(a0)
vsetvli zero, zero, e8, m1, tu, mu
vlseg5e8.v  v8,(a0)
vsetvli zero, zero, e16, mf4, tu, mu
vlseg5e16.v v8,(a0)
vsetvli zero, zero, e16, mf2, tu, mu
vlseg5e16.v v8,(a0)
vsetvli zero, zero, e16, m1, tu, mu
vlseg5e16.v v8,(a0)
vsetvli zero, zero, e32, mf2, tu, mu
vlseg5e32.v v8,(a0)
vsetvli zero, zero, e32, m1, tu, mu
vlseg5e32.v v8,(a0)
vsetvli zero, zero, e64, m1, tu, mu
vlseg5e64.v v8,(a0)

vsetvli zero, zero, e8, mf8, tu, mu
vlseg6e8.v  v8,(a0)
vsetvli zero, zero, e8, mf4, tu, mu
vlseg6e8.v  v8,(a0)
vsetvli zero, zero, e8, mf2, tu, mu
vlseg6e8.v  v8,(a0)
vsetvli zero, zero, e8, m1, tu, mu
vlseg6e8.v  v8,(a0)
vsetvli zero, zero, e16, mf4, tu, mu
vlseg6e16.v v8,(a0)
vsetvli zero, zero, e16, mf2, tu, mu
vlseg6e16.v v8,(a0)
vsetvli zero, zero, e16, m1, tu, mu
vlseg6e16.v v8,(a0)
vsetvli zero, zero, e32, mf2, tu, mu
vlseg6e32.v v8,(a0)
vsetvli zero, zero, e32, m1, tu, mu
vlseg6e32.v v8,(a0)
vsetvli zero, zero, e64, m1, tu, mu
vlseg6e64.v v8,(a0)

vsetvli zero, zero, e8, mf8, tu, mu
vlseg7e8.v  v8,(a0)
vsetvli zero, zero, e8, mf4, tu, mu
vlseg7e8.v  v8,(a0)
vsetvli zero, zero, e8, mf2, tu, mu
vlseg7e8.v  v8,(a0)
vsetvli zero, zero, e8, m1, tu, mu
vlseg7e8.v  v8,(a0)
vsetvli zero, zero, e16, mf4, tu, mu
vlseg7e16.v v8,(a0)
vsetvli zero, zero, e16, mf2, tu, mu
vlseg7e16.v v8,(a0)
vsetvli zero, zero, e16, m1, tu, mu
vlseg7e16.v v8,(a0)
vsetvli zero, zero, e32, mf2, tu, mu
vlseg7e32.v v8,(a0)
vsetvli zero, zero, e32, m1, tu, mu
vlseg7e32.v v8,(a0)
vsetvli zero, zero, e64, m1, tu, mu
vlseg7e64.v v8,(a0)

vsetvli zero, zero, e8, mf8, tu, mu
vlseg8e8.v  v8,(a0)
vsetvli zero, zero, e8, mf4, tu, mu
vlseg8e8.v  v8,(a0)
vsetvli zero, zero, e8, mf2, tu, mu
vlseg8e8.v  v8,(a0)
vsetvli zero, zero, e8, m1, tu, mu
vlseg8e8.v  v8,(a0)
vsetvli zero, zero, e16, mf4, tu, mu
vlseg8e16.v v8,(a0)
vsetvli zero, zero, e16, mf2, tu, mu
vlseg8e16.v v8,(a0)
vsetvli zero, zero, e16, m1, tu, mu
vlseg8e16.v v8,(a0)
vsetvli zero, zero, e32, mf2, tu, mu
vlseg8e32.v v8,(a0)
vsetvli zero, zero, e32, m1, tu, mu
vlseg8e32.v v8,(a0)
vsetvli zero, zero, e64, m1, tu, mu
vlseg8e64.v v8,(a0)

vsetvli zero, zero, e8, mf8, tu, mu
vsseg2e8.v  v8,(a0)
vsetvli zero, zero, e8, mf4, tu, mu
vsseg2e8.v  v8,(a0)
vsetvli zero, zero, e8, mf2, tu, mu
vsseg2e8.v  v8,(a0)
vsetvli zero, zero, e8, m1, tu, mu
vsseg2e8.v  v8,(a0)
vsetvli zero, zero, e8, m2, tu, mu
vsseg2e8.v  v8,(a0)
vsetvli zero, zero, e8, m4, tu, mu
vsseg2e8.v  v8,(a0)
vsetvli zero, zero, e16, mf4, tu, mu
vsseg2e16.v v8,(a0)
vsetvli zero, zero, e16, mf2, tu, mu
vsseg2e16.v v8,(a0)
vsetvli zero, zero, e16, m1, tu, mu
vsseg2e16.v v8,(a0)
vsetvli zero, zero, e16, m2, tu, mu
vsseg2e16.v v8,(a0)
vsetvli zero, zero, e16, m4, tu, mu
vsseg2e16.v v8,(a0)
vsetvli zero, zero, e32, mf2, tu, mu
vsseg2e32.v v8,(a0)
vsetvli zero, zero, e32, m1, tu, mu
vsseg2e32.v v8,(a0)
vsetvli zero, zero, e32, m2, tu, mu
vsseg2e32.v v8,(a0)
vsetvli zero, zero, e32, m4, tu, mu
vsseg2e32.v v8,(a0)
vsetvli zero, zero, e64, m1, tu, mu
vsseg2e64.v v8,(a0)
vsetvli zero, zero, e64, m2, tu, mu
vsseg2e64.v v8,(a0)
vsetvli zero, zero, e64, m4, tu, mu
vsseg2e64.v v8,(a0)

vsetvli zero, zero, e8, mf8, tu, mu
vsseg3e8.v  v8,(a0)
vsetvli zero, zero, e8, mf4, tu, mu
vsseg3e8.v  v8,(a0)
vsetvli zero, zero, e8, mf2, tu, mu
vsseg3e8.v  v8,(a0)
vsetvli zero, zero, e8, m1, tu, mu
vsseg3e8.v  v8,(a0)
vsetvli zero, zero, e8, m2, tu, mu
vsseg3e8.v  v8,(a0)
vsetvli zero, zero, e16, mf4, tu, mu
vsseg3e16.v v8,(a0)
vsetvli zero, zero, e16, mf2, tu, mu
vsseg3e16.v v8,(a0)
vsetvli zero, zero, e16, m1, tu, mu
vsseg3e16.v v8,(a0)
vsetvli zero, zero, e16, m2, tu, mu
vsseg3e16.v v8,(a0)
vsetvli zero, zero, e32, mf2, tu, mu
vsseg3e32.v v8,(a0)
vsetvli zero, zero, e32, m1, tu, mu
vsseg3e32.v v8,(a0)
vsetvli zero, zero, e32, m2, tu, mu
vsseg3e32.v v8,(a0)
vsetvli zero, zero, e64, m1, tu, mu
vsseg3e64.v v8,(a0)
vsetvli zero, zero, e64, m2, tu, mu
vsseg3e64.v v8,(a0)

vsetvli zero, zero, e8, mf8, tu, mu
vsseg4e8.v  v8,(a0)
vsetvli zero, zero, e8, mf4, tu, mu
vsseg4e8.v  v8,(a0)
vsetvli zero, zero, e8, mf2, tu, mu
vsseg4e8.v  v8,(a0)
vsetvli zero, zero, e8, m1, tu, mu
vsseg4e8.v  v8,(a0)
vsetvli zero, zero, e8, m2, tu, mu
vsseg4e8.v  v8,(a0)
vsetvli zero, zero, e16, mf4, tu, mu
vsseg4e16.v v8,(a0)
vsetvli zero, zero, e16, mf2, tu, mu
vsseg4e16.v v8,(a0)
vsetvli zero, zero, e16, m1, tu, mu
vsseg4e16.v v8,(a0)
vsetvli zero, zero, e16, m2, tu, mu
vsseg4e16.v v8,(a0)
vsetvli zero, zero, e32, mf2, tu, mu
vsseg4e32.v v8,(a0)
vsetvli zero, zero, e32, m1, tu, mu
vsseg4e32.v v8,(a0)
vsetvli zero, zero, e32, m2, tu, mu
vsseg4e32.v v8,(a0)
vsetvli zero, zero, e64, m1, tu, mu
vsseg4e64.v v8,(a0)
vsetvli zero, zero, e64, m2, tu, mu
vsseg4e64.v v8,(a0)

vsetvli zero, zero, e8, mf8, tu, mu
vsseg5e8.v  v8,(a0)
vsetvli zero, zero, e8, mf4, tu, mu
vsseg5e8.v  v8,(a0)
vsetvli zero, zero, e8, mf2, tu, mu
vsseg5e8.v  v8,(a0)
vsetvli zero, zero, e8, m1, tu, mu
vsseg5e8.v  v8,(a0)
vsetvli zero, zero, e16, mf4, tu, mu
vsseg5e16.v v8,(a0)
vsetvli zero, zero, e16, mf2, tu, mu
vsseg5e16.v v8,(a0)
vsetvli zero, zero, e16, m1, tu, mu
vsseg5e16.v v8,(a0)
vsetvli zero, zero, e32, mf2, tu, mu
vsseg5e32.v v8,(a0)
vsetvli zero, zero, e32, m1, tu, mu
vsseg5e32.v v8,(a0)
vsetvli zero, zero, e64, m1, tu, mu
vsseg5e64.v v8,(a0)

vsetvli zero, zero, e8, mf8, tu, mu
vsseg6e8.v  v8,(a0)
vsetvli zero, zero, e8, mf4, tu, mu
vsseg6e8.v  v8,(a0)
vsetvli zero, zero, e8, mf2, tu, mu
vsseg6e8.v  v8,(a0)
vsetvli zero, zero, e8, m1, tu, mu
vsseg6e8.v  v8,(a0)
vsetvli zero, zero, e16, mf4, tu, mu
vsseg6e16.v v8,(a0)
vsetvli zero, zero, e16, mf2, tu, mu
vsseg6e16.v v8,(a0)
vsetvli zero, zero, e16, m1, tu, mu
vsseg6e16.v v8,(a0)
vsetvli zero, zero, e32, mf2, tu, mu
vsseg6e32.v v8,(a0)
vsetvli zero, zero, e32, m1, tu, mu
vsseg6e32.v v8,(a0)
vsetvli zero, zero, e64, m1, tu, mu
vsseg6e64.v v8,(a0)

vsetvli zero, zero, e8, mf8, tu, mu
vsseg7e8.v  v8,(a0)
vsetvli zero, zero, e8, mf4, tu, mu
vsseg7e8.v  v8,(a0)
vsetvli zero, zero, e8, mf2, tu, mu
vsseg7e8.v  v8,(a0)
vsetvli zero, zero, e8, m1, tu, mu
vsseg7e8.v  v8,(a0)
vsetvli zero, zero, e16, mf4, tu, mu
vsseg7e16.v v8,(a0)
vsetvli zero, zero, e16, mf2, tu, mu
vsseg7e16.v v8,(a0)
vsetvli zero, zero, e16, m1, tu, mu
vsseg7e16.v v8,(a0)
vsetvli zero, zero, e32, mf2, tu, mu
vsseg7e32.v v8,(a0)
vsetvli zero, zero, e32, m1, tu, mu
vsseg7e32.v v8,(a0)
vsetvli zero, zero, e64, m1, tu, mu
vsseg7e64.v v8,(a0)

vsetvli zero, zero, e8, mf8, tu, mu
vsseg8e8.v  v8,(a0)
vsetvli zero, zero, e8, mf4, tu, mu
vsseg8e8.v  v8,(a0)
vsetvli zero, zero, e8, mf2, tu, mu
vsseg8e8.v  v8,(a0)
vsetvli zero, zero, e8, m1, tu, mu
vsseg8e8.v  v8,(a0)
vsetvli zero, zero, e16, mf4, tu, mu
vsseg8e16.v v8,(a0)
vsetvli zero, zero, e16, mf2, tu, mu
vsseg8e16.v v8,(a0)
vsetvli zero, zero, e16, m1, tu, mu
vsseg8e16.v v8,(a0)
vsetvli zero, zero, e32, mf2, tu, mu
vsseg8e32.v v8,(a0)
vsetvli zero, zero, e32, m1, tu, mu
vsseg8e32.v v8,(a0)
vsetvli zero, zero, e64, m1, tu, mu
vsseg8e64.v v8,(a0)

vsetvli zero, zero, e8, mf8, tu, mu
vlsseg2e8.v v8, (a0), a1
vsetvli zero, zero, e8, mf4, tu, mu
vlsseg2e8.v v8, (a0), a1
vsetvli zero, zero, e8, mf2, tu, mu
vlsseg2e8.v v8, (a0), a1
vsetvli zero, zero, e8, m1, tu, mu
vlsseg2e8.v v8, (a0), a1
vsetvli zero, zero, e8, m2, tu, mu
vlsseg2e8.v v8, (a0), a1
vsetvli zero, zero, e8, m4, tu, mu
vlsseg2e8.v v8, (a0), a1
vsetvli zero, zero, e16, mf4, tu, mu
vlsseg2e16.v  v8, (a0), a1
vsetvli zero, zero, e16, mf2, tu, mu
vlsseg2e16.v  v8, (a0), a1
vsetvli zero, zero, e16, m1, tu, mu
vlsseg2e16.v  v8, (a0), a1
vsetvli zero, zero, e16, m2, tu, mu
vlsseg2e16.v  v8, (a0), a1
vsetvli zero, zero, e16, m4, tu, mu
vlsseg2e16.v  v8, (a0), a1
vsetvli zero, zero, e32, mf2, tu, mu
vlsseg2e32.v  v8, (a0), a1
vsetvli zero, zero, e32, m1, tu, mu
vlsseg2e32.v  v8, (a0), a1
vsetvli zero, zero, e32, m2, tu, mu
vlsseg2e32.v  v8, (a0), a1
vsetvli zero, zero, e32, m4, tu, mu
vlsseg2e32.v  v8, (a0), a1
vsetvli zero, zero, e64, m1, tu, mu
vlsseg2e64.v  v8, (a0), a1
vsetvli zero, zero, e64, m2, tu, mu
vlsseg2e64.v  v8, (a0), a1
vsetvli zero, zero, e64, m4, tu, mu
vlsseg2e64.v  v8, (a0), a1

vsetvli zero, zero, e8, mf8, tu, mu
vlsseg3e8.v v8, (a0), a1
vsetvli zero, zero, e8, mf4, tu, mu
vlsseg3e8.v v8, (a0), a1
vsetvli zero, zero, e8, mf2, tu, mu
vlsseg3e8.v v8, (a0), a1
vsetvli zero, zero, e8, m1, tu, mu
vlsseg3e8.v v8, (a0), a1
vsetvli zero, zero, e8, m2, tu, mu
vlsseg3e8.v v8, (a0), a1
vsetvli zero, zero, e16, mf4, tu, mu
vlsseg3e16.v  v8, (a0), a1
vsetvli zero, zero, e16, mf2, tu, mu
vlsseg3e16.v  v8, (a0), a1
vsetvli zero, zero, e16, m1, tu, mu
vlsseg3e16.v  v8, (a0), a1
vsetvli zero, zero, e16, m2, tu, mu
vlsseg3e16.v  v8, (a0), a1
vsetvli zero, zero, e32, mf2, tu, mu
vlsseg3e32.v  v8, (a0), a1
vsetvli zero, zero, e32, m1, tu, mu
vlsseg3e32.v  v8, (a0), a1
vsetvli zero, zero, e32, m2, tu, mu
vlsseg3e32.v  v8, (a0), a1
vsetvli zero, zero, e64, m1, tu, mu
vlsseg3e64.v  v8, (a0), a1
vsetvli zero, zero, e64, m2, tu, mu
vlsseg3e64.v  v8, (a0), a1

vsetvli zero, zero, e8, mf8, tu, mu
vlsseg4e8.v v8, (a0), a1
vsetvli zero, zero, e8, mf4, tu, mu
vlsseg4e8.v v8, (a0), a1
vsetvli zero, zero, e8, mf2, tu, mu
vlsseg4e8.v v8, (a0), a1
vsetvli zero, zero, e8, m1, tu, mu
vlsseg4e8.v v8, (a0), a1
vsetvli zero, zero, e8, m2, tu, mu
vlsseg4e8.v v8, (a0), a1
vsetvli zero, zero, e16, mf4, tu, mu
vlsseg4e16.v  v8, (a0), a1
vsetvli zero, zero, e16, mf2, tu, mu
vlsseg4e16.v  v8, (a0), a1
vsetvli zero, zero, e16, m1, tu, mu
vlsseg4e16.v  v8, (a0), a1
vsetvli zero, zero, e16, m2, tu, mu
vlsseg4e16.v  v8, (a0), a1
vsetvli zero, zero, e32, mf2, tu, mu
vlsseg4e32.v  v8, (a0), a1
vsetvli zero, zero, e32, m1, tu, mu
vlsseg4e32.v  v8, (a0), a1
vsetvli zero, zero, e32, m2, tu, mu
vlsseg4e32.v  v8, (a0), a1
vsetvli zero, zero, e64, m1, tu, mu
vlsseg4e64.v  v8, (a0), a1
vsetvli zero, zero, e64, m2, tu, mu
vlsseg4e64.v  v8, (a0), a1

vsetvli zero, zero, e8, mf8, tu, mu
vlsseg5e8.v v8, (a0), a1
vsetvli zero, zero, e8, mf4, tu, mu
vlsseg5e8.v v8, (a0), a1
vsetvli zero, zero, e8, mf2, tu, mu
vlsseg5e8.v v8, (a0), a1
vsetvli zero, zero, e8, m1, tu, mu
vlsseg5e8.v v8, (a0), a1
vsetvli zero, zero, e16, mf4, tu, mu
vlsseg5e16.v  v8, (a0), a1
vsetvli zero, zero, e16, mf2, tu, mu
vlsseg5e16.v  v8, (a0), a1
vsetvli zero, zero, e16, m1, tu, mu
vlsseg5e16.v  v8, (a0), a1
vsetvli zero, zero, e32, mf2, tu, mu
vlsseg5e32.v  v8, (a0), a1
vsetvli zero, zero, e32, m1, tu, mu
vlsseg5e32.v  v8, (a0), a1
vsetvli zero, zero, e64, m1, tu, mu
vlsseg5e64.v  v8, (a0), a1

vsetvli zero, zero, e8, mf8, tu, mu
vlsseg6e8.v v8, (a0), a1
vsetvli zero, zero, e8, mf4, tu, mu
vlsseg6e8.v v8, (a0), a1
vsetvli zero, zero, e8, mf2, tu, mu
vlsseg6e8.v v8, (a0), a1
vsetvli zero, zero, e8, m1, tu, mu
vlsseg6e8.v v8, (a0), a1
vsetvli zero, zero, e16, mf4, tu, mu
vlsseg6e16.v  v8, (a0), a1
vsetvli zero, zero, e16, mf2, tu, mu
vlsseg6e16.v  v8, (a0), a1
vsetvli zero, zero, e16, m1, tu, mu
vlsseg6e16.v  v8, (a0), a1
vsetvli zero, zero, e32, mf2, tu, mu
vlsseg6e32.v  v8, (a0), a1
vsetvli zero, zero, e32, m1, tu, mu
vlsseg6e32.v  v8, (a0), a1
vsetvli zero, zero, e64, m1, tu, mu
vlsseg6e64.v  v8, (a0), a1

vsetvli zero, zero, e8, mf8, tu, mu
vlsseg7e8.v v8, (a0), a1
vsetvli zero, zero, e8, mf4, tu, mu
vlsseg7e8.v v8, (a0), a1
vsetvli zero, zero, e8, mf2, tu, mu
vlsseg7e8.v v8, (a0), a1
vsetvli zero, zero, e8, m1, tu, mu
vlsseg7e8.v v8, (a0), a1
vsetvli zero, zero, e16, mf4, tu, mu
vlsseg7e16.v  v8, (a0), a1
vsetvli zero, zero, e16, mf2, tu, mu
vlsseg7e16.v  v8, (a0), a1
vsetvli zero, zero, e16, m1, tu, mu
vlsseg7e16.v  v8, (a0), a1
vsetvli zero, zero, e32, mf2, tu, mu
vlsseg7e32.v  v8, (a0), a1
vsetvli zero, zero, e32, m1, tu, mu
vlsseg7e32.v  v8, (a0), a1
vsetvli zero, zero, e64, m1, tu, mu
vlsseg7e64.v  v8, (a0), a1

vsetvli zero, zero, e8, mf8, tu, mu
vlsseg8e8.v v8, (a0), a1
vsetvli zero, zero, e8, mf4, tu, mu
vlsseg8e8.v v8, (a0), a1
vsetvli zero, zero, e8, mf2, tu, mu
vlsseg8e8.v v8, (a0), a1
vsetvli zero, zero, e8, m1, tu, mu
vlsseg8e8.v v8, (a0), a1
vsetvli zero, zero, e16, mf4, tu, mu
vlsseg8e16.v  v8, (a0), a1
vsetvli zero, zero, e16, mf2, tu, mu
vlsseg8e16.v  v8, (a0), a1
vsetvli zero, zero, e16, m1, tu, mu
vlsseg8e16.v  v8, (a0), a1
vsetvli zero, zero, e32, mf2, tu, mu
vlsseg8e32.v  v8, (a0), a1
vsetvli zero, zero, e32, m1, tu, mu
vlsseg8e32.v  v8, (a0), a1
vsetvli zero, zero, e64, m1, tu, mu
vlsseg8e64.v  v8, (a0), a1

vsetvli zero, zero, e8, mf8, tu, mu
vssseg2e8.v v8, (a0), a1
vsetvli zero, zero, e8, mf4, tu, mu
vssseg2e8.v v8, (a0), a1
vsetvli zero, zero, e8, mf2, tu, mu
vssseg2e8.v v8, (a0), a1
vsetvli zero, zero, e8, m1, tu, mu
vssseg2e8.v v8, (a0), a1
vsetvli zero, zero, e8, m2, tu, mu
vssseg2e8.v v8, (a0), a1
vsetvli zero, zero, e8, m4, tu, mu
vssseg2e8.v v8, (a0), a1
vsetvli zero, zero, e16, mf4, tu, mu
vssseg2e16.v  v8, (a0), a1
vsetvli zero, zero, e16, mf2, tu, mu
vssseg2e16.v  v8, (a0), a1
vsetvli zero, zero, e16, m1, tu, mu
vssseg2e16.v  v8, (a0), a1
vsetvli zero, zero, e16, m2, tu, mu
vssseg2e16.v  v8, (a0), a1
vsetvli zero, zero, e16, m4, tu, mu
vssseg2e16.v  v8, (a0), a1
vsetvli zero, zero, e32, mf2, tu, mu
vssseg2e32.v  v8, (a0), a1
vsetvli zero, zero, e32, m1, tu, mu
vssseg2e32.v  v8, (a0), a1
vsetvli zero, zero, e32, m2, tu, mu
vssseg2e32.v  v8, (a0), a1
vsetvli zero, zero, e32, m4, tu, mu
vssseg2e32.v  v8, (a0), a1
vsetvli zero, zero, e64, m1, tu, mu
vssseg2e64.v  v8, (a0), a1
vsetvli zero, zero, e64, m2, tu, mu
vssseg2e64.v  v8, (a0), a1
vsetvli zero, zero, e64, m4, tu, mu
vssseg2e64.v  v8, (a0), a1

vsetvli zero, zero, e8, mf8, tu, mu
vssseg3e8.v v8, (a0), a1
vsetvli zero, zero, e8, mf4, tu, mu
vssseg3e8.v v8, (a0), a1
vsetvli zero, zero, e8, mf2, tu, mu
vssseg3e8.v v8, (a0), a1
vsetvli zero, zero, e8, m1, tu, mu
vssseg3e8.v v8, (a0), a1
vsetvli zero, zero, e8, m2, tu, mu
vssseg3e8.v v8, (a0), a1
vsetvli zero, zero, e16, mf4, tu, mu
vssseg3e16.v  v8, (a0), a1
vsetvli zero, zero, e16, mf2, tu, mu
vssseg3e16.v  v8, (a0), a1
vsetvli zero, zero, e16, m1, tu, mu
vssseg3e16.v  v8, (a0), a1
vsetvli zero, zero, e16, m2, tu, mu
vssseg3e16.v  v8, (a0), a1
vsetvli zero, zero, e32, mf2, tu, mu
vssseg3e32.v  v8, (a0), a1
vsetvli zero, zero, e32, m1, tu, mu
vssseg3e32.v  v8, (a0), a1
vsetvli zero, zero, e32, m2, tu, mu
vssseg3e32.v  v8, (a0), a1
vsetvli zero, zero, e64, m1, tu, mu
vssseg3e64.v  v8, (a0), a1
vsetvli zero, zero, e64, m2, tu, mu
vssseg3e64.v  v8, (a0), a1

vsetvli zero, zero, e8, mf8, tu, mu
vssseg4e8.v v8, (a0), a1
vsetvli zero, zero, e8, mf4, tu, mu
vssseg4e8.v v8, (a0), a1
vsetvli zero, zero, e8, mf2, tu, mu
vssseg4e8.v v8, (a0), a1
vsetvli zero, zero, e8, m1, tu, mu
vssseg4e8.v v8, (a0), a1
vsetvli zero, zero, e8, m2, tu, mu
vssseg4e8.v v8, (a0), a1
vsetvli zero, zero, e16, mf4, tu, mu
vssseg4e16.v  v8, (a0), a1
vsetvli zero, zero, e16, mf2, tu, mu
vssseg4e16.v  v8, (a0), a1
vsetvli zero, zero, e16, m1, tu, mu
vssseg4e16.v  v8, (a0), a1
vsetvli zero, zero, e16, m2, tu, mu
vssseg4e16.v  v8, (a0), a1
vsetvli zero, zero, e32, mf2, tu, mu
vssseg4e32.v  v8, (a0), a1
vsetvli zero, zero, e32, m1, tu, mu
vssseg4e32.v  v8, (a0), a1
vsetvli zero, zero, e32, m2, tu, mu
vssseg4e32.v  v8, (a0), a1
vsetvli zero, zero, e64, m1, tu, mu
vssseg4e64.v  v8, (a0), a1
vsetvli zero, zero, e64, m2, tu, mu
vssseg4e64.v  v8, (a0), a1

vsetvli zero, zero, e8, mf8, tu, mu
vssseg5e8.v v8, (a0), a1
vsetvli zero, zero, e8, mf4, tu, mu
vssseg5e8.v v8, (a0), a1
vsetvli zero, zero, e8, mf2, tu, mu
vssseg5e8.v v8, (a0), a1
vsetvli zero, zero, e8, m1, tu, mu
vssseg5e8.v v8, (a0), a1
vsetvli zero, zero, e16, mf4, tu, mu
vssseg5e16.v  v8, (a0), a1
vsetvli zero, zero, e16, mf2, tu, mu
vssseg5e16.v  v8, (a0), a1
vsetvli zero, zero, e16, m1, tu, mu
vssseg5e16.v  v8, (a0), a1
vsetvli zero, zero, e32, mf2, tu, mu
vssseg5e32.v  v8, (a0), a1
vsetvli zero, zero, e32, m1, tu, mu
vssseg5e32.v  v8, (a0), a1
vsetvli zero, zero, e64, m1, tu, mu
vssseg5e64.v  v8, (a0), a1

vsetvli zero, zero, e8, mf8, tu, mu
vssseg6e8.v v8, (a0), a1
vsetvli zero, zero, e8, mf4, tu, mu
vssseg6e8.v v8, (a0), a1
vsetvli zero, zero, e8, mf2, tu, mu
vssseg6e8.v v8, (a0), a1
vsetvli zero, zero, e8, m1, tu, mu
vssseg6e8.v v8, (a0), a1
vsetvli zero, zero, e16, mf4, tu, mu
vssseg6e16.v  v8, (a0), a1
vsetvli zero, zero, e16, mf2, tu, mu
vssseg6e16.v  v8, (a0), a1
vsetvli zero, zero, e16, m1, tu, mu
vssseg6e16.v  v8, (a0), a1
vsetvli zero, zero, e32, mf2, tu, mu
vssseg6e32.v  v8, (a0), a1
vsetvli zero, zero, e32, m1, tu, mu
vssseg6e32.v  v8, (a0), a1
vsetvli zero, zero, e64, m1, tu, mu
vssseg6e64.v  v8, (a0), a1

vsetvli zero, zero, e8, mf8, tu, mu
vssseg7e8.v v8, (a0), a1
vsetvli zero, zero, e8, mf4, tu, mu
vssseg7e8.v v8, (a0), a1
vsetvli zero, zero, e8, mf2, tu, mu
vssseg7e8.v v8, (a0), a1
vsetvli zero, zero, e8, m1, tu, mu
vssseg7e8.v v8, (a0), a1
vsetvli zero, zero, e16, mf4, tu, mu
vssseg7e16.v  v8, (a0), a1
vsetvli zero, zero, e16, mf2, tu, mu
vssseg7e16.v  v8, (a0), a1
vsetvli zero, zero, e16, m1, tu, mu
vssseg7e16.v  v8, (a0), a1
vsetvli zero, zero, e32, mf2, tu, mu
vssseg7e32.v  v8, (a0), a1
vsetvli zero, zero, e32, m1, tu, mu
vssseg7e32.v  v8, (a0), a1
vsetvli zero, zero, e64, m1, tu, mu
vssseg7e64.v  v8, (a0), a1

vsetvli zero, zero, e8, mf8, tu, mu
vssseg8e8.v v8, (a0), a1
vsetvli zero, zero, e8, mf4, tu, mu
vssseg8e8.v v8, (a0), a1
vsetvli zero, zero, e8, mf2, tu, mu
vssseg8e8.v v8, (a0), a1
vsetvli zero, zero, e8, m1, tu, mu
vssseg8e8.v v8, (a0), a1
vsetvli zero, zero, e16, mf4, tu, mu
vssseg8e16.v  v8, (a0), a1
vsetvli zero, zero, e16, mf2, tu, mu
vssseg8e16.v  v8, (a0), a1
vsetvli zero, zero, e16, m1, tu, mu
vssseg8e16.v  v8, (a0), a1
vsetvli zero, zero, e32, mf2, tu, mu
vssseg8e32.v  v8, (a0), a1
vsetvli zero, zero, e32, m1, tu, mu
vssseg8e32.v  v8, (a0), a1
vsetvli zero, zero, e64, m1, tu, mu
vssseg8e64.v  v8, (a0), a1

vsetvli zero, zero, e8, mf8, tu, mu
vlseg2e8ff.v  v8,(a0)
vsetvli zero, zero, e8, mf4, tu, mu
vlseg2e8ff.v  v8,(a0)
vsetvli zero, zero, e8, mf2, tu, mu
vlseg2e8ff.v  v8,(a0)
vsetvli zero, zero, e8, m1, tu, mu
vlseg2e8ff.v  v8,(a0)
vsetvli zero, zero, e8, m2, tu, mu
vlseg2e8ff.v  v8,(a0)
vsetvli zero, zero, e8, m4, tu, mu
vlseg2e8ff.v  v8,(a0)
vsetvli zero, zero, e16, mf4, tu, mu
vlseg2e16ff.v v8,(a0)
vsetvli zero, zero, e16, mf2, tu, mu
vlseg2e16ff.v v8,(a0)
vsetvli zero, zero, e16, m1, tu, mu
vlseg2e16ff.v v8,(a0)
vsetvli zero, zero, e16, m2, tu, mu
vlseg2e16ff.v v8,(a0)
vsetvli zero, zero, e16, m4, tu, mu
vlseg2e16ff.v v8,(a0)
vsetvli zero, zero, e32, mf2, tu, mu
vlseg2e32ff.v v8,(a0)
vsetvli zero, zero, e32, m1, tu, mu
vlseg2e32ff.v v8,(a0)
vsetvli zero, zero, e32, m2, tu, mu
vlseg2e32ff.v v8,(a0)
vsetvli zero, zero, e32, m4, tu, mu
vlseg2e32ff.v v8,(a0)
vsetvli zero, zero, e64, m1, tu, mu
vlseg2e64ff.v v8,(a0)
vsetvli zero, zero, e64, m2, tu, mu
vlseg2e64ff.v v8,(a0)
vsetvli zero, zero, e64, m4, tu, mu
vlseg2e64ff.v v8,(a0)

vsetvli zero, zero, e8, mf8, tu, mu
vlseg3e8ff.v  v8,(a0)
vsetvli zero, zero, e8, mf4, tu, mu
vlseg3e8ff.v  v8,(a0)
vsetvli zero, zero, e8, mf2, tu, mu
vlseg3e8ff.v  v8,(a0)
vsetvli zero, zero, e8, m1, tu, mu
vlseg3e8ff.v  v8,(a0)
vsetvli zero, zero, e8, m2, tu, mu
vlseg3e8ff.v  v8,(a0)
vsetvli zero, zero, e16, mf4, tu, mu
vlseg3e16ff.v v8,(a0)
vsetvli zero, zero, e16, mf2, tu, mu
vlseg3e16ff.v v8,(a0)
vsetvli zero, zero, e16, m1, tu, mu
vlseg3e16ff.v v8,(a0)
vsetvli zero, zero, e16, m2, tu, mu
vlseg3e16ff.v v8,(a0)
vsetvli zero, zero, e32, mf2, tu, mu
vlseg3e32ff.v v8,(a0)
vsetvli zero, zero, e32, m1, tu, mu
vlseg3e32ff.v v8,(a0)
vsetvli zero, zero, e32, m2, tu, mu
vlseg3e32ff.v v8,(a0)
vsetvli zero, zero, e64, m1, tu, mu
vlseg3e64ff.v v8,(a0)
vsetvli zero, zero, e64, m2, tu, mu
vlseg3e64ff.v v8,(a0)

vsetvli zero, zero, e8, mf8, tu, mu
vlseg4e8ff.v  v8,(a0)
vsetvli zero, zero, e8, mf4, tu, mu
vlseg4e8ff.v  v8,(a0)
vsetvli zero, zero, e8, mf2, tu, mu
vlseg4e8ff.v  v8,(a0)
vsetvli zero, zero, e8, m1, tu, mu
vlseg4e8ff.v  v8,(a0)
vsetvli zero, zero, e8, m2, tu, mu
vlseg4e8ff.v  v8,(a0)
vsetvli zero, zero, e16, mf4, tu, mu
vlseg4e16ff.v v8,(a0)
vsetvli zero, zero, e16, mf2, tu, mu
vlseg4e16ff.v v8,(a0)
vsetvli zero, zero, e16, m1, tu, mu
vlseg4e16ff.v v8,(a0)
vsetvli zero, zero, e16, m2, tu, mu
vlseg4e16ff.v v8,(a0)
vsetvli zero, zero, e32, mf2, tu, mu
vlseg4e32ff.v v8,(a0)
vsetvli zero, zero, e32, m1, tu, mu
vlseg4e32ff.v v8,(a0)
vsetvli zero, zero, e32, m2, tu, mu
vlseg4e32ff.v v8,(a0)
vsetvli zero, zero, e64, m1, tu, mu
vlseg4e64ff.v v8,(a0)
vsetvli zero, zero, e64, m2, tu, mu
vlseg4e64ff.v v8,(a0)

vsetvli zero, zero, e8, mf8, tu, mu
vlseg5e8ff.v  v8,(a0)
vsetvli zero, zero, e8, mf4, tu, mu
vlseg5e8ff.v  v8,(a0)
vsetvli zero, zero, e8, mf2, tu, mu
vlseg5e8ff.v  v8,(a0)
vsetvli zero, zero, e8, m1, tu, mu
vlseg5e8ff.v  v8,(a0)
vsetvli zero, zero, e16, mf4, tu, mu
vlseg5e16ff.v v8,(a0)
vsetvli zero, zero, e16, mf2, tu, mu
vlseg5e16ff.v v8,(a0)
vsetvli zero, zero, e16, m1, tu, mu
vlseg5e16ff.v v8,(a0)
vsetvli zero, zero, e32, mf2, tu, mu
vlseg5e32ff.v v8,(a0)
vsetvli zero, zero, e32, m1, tu, mu
vlseg5e32ff.v v8,(a0)
vsetvli zero, zero, e64, m1, tu, mu
vlseg5e64ff.v v8,(a0)

vsetvli zero, zero, e8, mf8, tu, mu
vlseg6e8ff.v  v8,(a0)
vsetvli zero, zero, e8, mf4, tu, mu
vlseg6e8ff.v  v8,(a0)
vsetvli zero, zero, e8, mf2, tu, mu
vlseg6e8ff.v  v8,(a0)
vsetvli zero, zero, e8, m1, tu, mu
vlseg6e8ff.v  v8,(a0)
vsetvli zero, zero, e16, mf4, tu, mu
vlseg6e16ff.v v8,(a0)
vsetvli zero, zero, e16, mf2, tu, mu
vlseg6e16ff.v v8,(a0)
vsetvli zero, zero, e16, m1, tu, mu
vlseg6e16ff.v v8,(a0)
vsetvli zero, zero, e32, mf2, tu, mu
vlseg6e32ff.v v8,(a0)
vsetvli zero, zero, e32, m1, tu, mu
vlseg6e32ff.v v8,(a0)
vsetvli zero, zero, e64, m1, tu, mu
vlseg6e64ff.v v8,(a0)

vsetvli zero, zero, e8, mf8, tu, mu
vlseg7e8ff.v  v8,(a0)
vsetvli zero, zero, e8, mf4, tu, mu
vlseg7e8ff.v  v8,(a0)
vsetvli zero, zero, e8, mf2, tu, mu
vlseg7e8ff.v  v8,(a0)
vsetvli zero, zero, e8, m1, tu, mu
vlseg7e8ff.v  v8,(a0)
vsetvli zero, zero, e16, mf4, tu, mu
vlseg7e16ff.v v8,(a0)
vsetvli zero, zero, e16, mf2, tu, mu
vlseg7e16ff.v v8,(a0)
vsetvli zero, zero, e16, m1, tu, mu
vlseg7e16ff.v v8,(a0)
vsetvli zero, zero, e32, mf2, tu, mu
vlseg7e32ff.v v8,(a0)
vsetvli zero, zero, e32, m1, tu, mu
vlseg7e32ff.v v8,(a0)
vsetvli zero, zero, e64, m1, tu, mu
vlseg7e64ff.v v8,(a0)

vsetvli zero, zero, e8, mf8, tu, mu
vlseg8e8ff.v  v8,(a0)
vsetvli zero, zero, e8, mf4, tu, mu
vlseg8e8ff.v  v8,(a0)
vsetvli zero, zero, e8, mf2, tu, mu
vlseg8e8ff.v  v8,(a0)
vsetvli zero, zero, e8, m1, tu, mu
vlseg8e8ff.v  v8,(a0)
vsetvli zero, zero, e16, mf4, tu, mu
vlseg8e16ff.v v8,(a0)
vsetvli zero, zero, e16, mf2, tu, mu
vlseg8e16ff.v v8,(a0)
vsetvli zero, zero, e16, m1, tu, mu
vlseg8e16ff.v v8,(a0)
vsetvli zero, zero, e32, mf2, tu, mu
vlseg8e32ff.v v8,(a0)
vsetvli zero, zero, e32, m1, tu, mu
vlseg8e32ff.v v8,(a0)
vsetvli zero, zero, e64, m1, tu, mu
vlseg8e64ff.v v8,(a0)

vsetvli zero, zero, e8, mf8, tu, mu
vluxseg2ei8.v v8, (a0), v16
vsetvli zero, zero, e8, mf4, tu, mu
vluxseg2ei8.v v8, (a0), v16
vsetvli zero, zero, e8, mf2, tu, mu
vluxseg2ei8.v v8, (a0), v16
vsetvli zero, zero, e8, m1, tu, mu
vluxseg2ei8.v v8, (a0), v16
vsetvli zero, zero, e8, m2, tu, mu
vluxseg2ei8.v v8, (a0), v16
vsetvli zero, zero, e8, m4, tu, mu
vluxseg2ei8.v v8, (a0), v16
vsetvli zero, zero, e16, mf4, tu, mu
vluxseg2ei16.v  v8, (a0), v16
vsetvli zero, zero, e16, mf2, tu, mu
vluxseg2ei16.v  v8, (a0), v16
vsetvli zero, zero, e16, m1, tu, mu
vluxseg2ei16.v  v8, (a0), v16
vsetvli zero, zero, e16, m2, tu, mu
vluxseg2ei16.v  v8, (a0), v16
vsetvli zero, zero, e16, m4, tu, mu
vluxseg2ei16.v  v8, (a0), v16
vsetvli zero, zero, e32, mf2, tu, mu
vluxseg2ei32.v  v8, (a0), v16
vsetvli zero, zero, e32, m1, tu, mu
vluxseg2ei32.v  v8, (a0), v16
vsetvli zero, zero, e32, m2, tu, mu
vluxseg2ei32.v  v8, (a0), v16
vsetvli zero, zero, e32, m4, tu, mu
vluxseg2ei32.v  v8, (a0), v16
vsetvli zero, zero, e64, m1, tu, mu
vluxseg2ei64.v  v8, (a0), v16
vsetvli zero, zero, e64, m2, tu, mu
vluxseg2ei64.v  v8, (a0), v16
vsetvli zero, zero, e64, m4, tu, mu
vluxseg2ei64.v  v8, (a0), v16

vsetvli zero, zero, e8, mf8, tu, mu
vluxseg3ei8.v v8, (a0), v16
vsetvli zero, zero, e8, mf4, tu, mu
vluxseg3ei8.v v8, (a0), v16
vsetvli zero, zero, e8, mf2, tu, mu
vluxseg3ei8.v v8, (a0), v16
vsetvli zero, zero, e8, m1, tu, mu
vluxseg3ei8.v v8, (a0), v16
vsetvli zero, zero, e8, m2, tu, mu
vluxseg3ei8.v v8, (a0), v16
vsetvli zero, zero, e16, mf4, tu, mu
vluxseg3ei16.v  v8, (a0), v16
vsetvli zero, zero, e16, mf2, tu, mu
vluxseg3ei16.v  v8, (a0), v16
vsetvli zero, zero, e16, m1, tu, mu
vluxseg3ei16.v  v8, (a0), v16
vsetvli zero, zero, e16, m2, tu, mu
vluxseg3ei16.v  v8, (a0), v16
vsetvli zero, zero, e32, mf2, tu, mu
vluxseg3ei32.v  v8, (a0), v16
vsetvli zero, zero, e32, m1, tu, mu
vluxseg3ei32.v  v8, (a0), v16
vsetvli zero, zero, e32, m2, tu, mu
vluxseg3ei32.v  v8, (a0), v16
vsetvli zero, zero, e64, m1, tu, mu
vluxseg3ei64.v  v8, (a0), v16
vsetvli zero, zero, e64, m2, tu, mu
vluxseg3ei64.v  v8, (a0), v16

vsetvli zero, zero, e8, mf8, tu, mu
vluxseg4ei8.v v8, (a0), v16
vsetvli zero, zero, e8, mf4, tu, mu
vluxseg4ei8.v v8, (a0), v16
vsetvli zero, zero, e8, mf2, tu, mu
vluxseg4ei8.v v8, (a0), v16
vsetvli zero, zero, e8, m1, tu, mu
vluxseg4ei8.v v8, (a0), v16
vsetvli zero, zero, e8, m2, tu, mu
vluxseg4ei8.v v8, (a0), v16
vsetvli zero, zero, e16, mf4, tu, mu
vluxseg4ei16.v  v8, (a0), v16
vsetvli zero, zero, e16, mf2, tu, mu
vluxseg4ei16.v  v8, (a0), v16
vsetvli zero, zero, e16, m1, tu, mu
vluxseg4ei16.v  v8, (a0), v16
vsetvli zero, zero, e16, m2, tu, mu
vluxseg4ei16.v  v8, (a0), v16
vsetvli zero, zero, e32, mf2, tu, mu
vluxseg4ei32.v  v8, (a0), v16
vsetvli zero, zero, e32, m1, tu, mu
vluxseg4ei32.v  v8, (a0), v16
vsetvli zero, zero, e32, m2, tu, mu
vluxseg4ei32.v  v8, (a0), v16
vsetvli zero, zero, e64, m1, tu, mu
vluxseg4ei64.v  v8, (a0), v16
vsetvli zero, zero, e64, m2, tu, mu
vluxseg4ei64.v  v8, (a0), v16

vsetvli zero, zero, e8, mf8, tu, mu
vluxseg5ei8.v v8, (a0), v16
vsetvli zero, zero, e8, mf4, tu, mu
vluxseg5ei8.v v8, (a0), v16
vsetvli zero, zero, e8, mf2, tu, mu
vluxseg5ei8.v v8, (a0), v16
vsetvli zero, zero, e8, m1, tu, mu
vluxseg5ei8.v v8, (a0), v16
vsetvli zero, zero, e16, mf4, tu, mu
vluxseg5ei16.v  v8, (a0), v16
vsetvli zero, zero, e16, mf2, tu, mu
vluxseg5ei16.v  v8, (a0), v16
vsetvli zero, zero, e16, m1, tu, mu
vluxseg5ei16.v  v8, (a0), v16
vsetvli zero, zero, e32, mf2, tu, mu
vluxseg5ei32.v  v8, (a0), v16
vsetvli zero, zero, e32, m1, tu, mu
vluxseg5ei32.v  v8, (a0), v16
vsetvli zero, zero, e64, m1, tu, mu
vluxseg5ei64.v  v8, (a0), v16

vsetvli zero, zero, e8, mf8, tu, mu
vluxseg6ei8.v v8, (a0), v16
vsetvli zero, zero, e8, mf4, tu, mu
vluxseg6ei8.v v8, (a0), v16
vsetvli zero, zero, e8, mf2, tu, mu
vluxseg6ei8.v v8, (a0), v16
vsetvli zero, zero, e8, m1, tu, mu
vluxseg6ei8.v v8, (a0), v16
vsetvli zero, zero, e16, mf4, tu, mu
vluxseg6ei16.v  v8, (a0), v16
vsetvli zero, zero, e16, mf2, tu, mu
vluxseg6ei16.v  v8, (a0), v16
vsetvli zero, zero, e16, m1, tu, mu
vluxseg6ei16.v  v8, (a0), v16
vsetvli zero, zero, e32, mf2, tu, mu
vluxseg6ei32.v  v8, (a0), v16
vsetvli zero, zero, e32, m1, tu, mu
vluxseg6ei32.v  v8, (a0), v16
vsetvli zero, zero, e64, m1, tu, mu
vluxseg6ei64.v  v8, (a0), v16

vsetvli zero, zero, e8, mf8, tu, mu
vluxseg7ei8.v v8, (a0), v16
vsetvli zero, zero, e8, mf4, tu, mu
vluxseg7ei8.v v8, (a0), v16
vsetvli zero, zero, e8, mf2, tu, mu
vluxseg7ei8.v v8, (a0), v16
vsetvli zero, zero, e8, m1, tu, mu
vluxseg7ei8.v v8, (a0), v16
vsetvli zero, zero, e16, mf4, tu, mu
vluxseg7ei16.v  v8, (a0), v16
vsetvli zero, zero, e16, mf2, tu, mu
vluxseg7ei16.v  v8, (a0), v16
vsetvli zero, zero, e16, m1, tu, mu
vluxseg7ei16.v  v8, (a0), v16
vsetvli zero, zero, e32, mf2, tu, mu
vluxseg7ei32.v  v8, (a0), v16
vsetvli zero, zero, e32, m1, tu, mu
vluxseg7ei32.v  v8, (a0), v16
vsetvli zero, zero, e64, m1, tu, mu
vluxseg7ei64.v  v8, (a0), v16

vsetvli zero, zero, e8, mf8, tu, mu
vluxseg8ei8.v v8, (a0), v16
vsetvli zero, zero, e8, mf4, tu, mu
vluxseg8ei8.v v8, (a0), v16
vsetvli zero, zero, e8, mf2, tu, mu
vluxseg8ei8.v v8, (a0), v16
vsetvli zero, zero, e8, m1, tu, mu
vluxseg8ei8.v v8, (a0), v16
vsetvli zero, zero, e16, mf4, tu, mu
vluxseg8ei16.v  v8, (a0), v16
vsetvli zero, zero, e16, mf2, tu, mu
vluxseg8ei16.v  v8, (a0), v16
vsetvli zero, zero, e16, m1, tu, mu
vluxseg8ei16.v  v8, (a0), v16
vsetvli zero, zero, e32, mf2, tu, mu
vluxseg8ei32.v  v8, (a0), v16
vsetvli zero, zero, e32, m1, tu, mu
vluxseg8ei32.v  v8, (a0), v16
vsetvli zero, zero, e64, m1, tu, mu
vluxseg8ei64.v  v8, (a0), v16

vsetvli zero, zero, e8, mf8, tu, mu
vloxseg2ei8.v v8, (a0), v16
vsetvli zero, zero, e8, mf4, tu, mu
vloxseg2ei8.v v8, (a0), v16
vsetvli zero, zero, e8, mf2, tu, mu
vloxseg2ei8.v v8, (a0), v16
vsetvli zero, zero, e8, m1, tu, mu
vloxseg2ei8.v v8, (a0), v16
vsetvli zero, zero, e8, m2, tu, mu
vloxseg2ei8.v v8, (a0), v16
vsetvli zero, zero, e8, m4, tu, mu
vloxseg2ei8.v v8, (a0), v16
vsetvli zero, zero, e16, mf4, tu, mu
vloxseg2ei16.v  v8, (a0), v16
vsetvli zero, zero, e16, mf2, tu, mu
vloxseg2ei16.v  v8, (a0), v16
vsetvli zero, zero, e16, m1, tu, mu
vloxseg2ei16.v  v8, (a0), v16
vsetvli zero, zero, e16, m2, tu, mu
vloxseg2ei16.v  v8, (a0), v16
vsetvli zero, zero, e16, m4, tu, mu
vloxseg2ei16.v  v8, (a0), v16
vsetvli zero, zero, e32, mf2, tu, mu
vloxseg2ei32.v  v8, (a0), v16
vsetvli zero, zero, e32, m1, tu, mu
vloxseg2ei32.v  v8, (a0), v16
vsetvli zero, zero, e32, m2, tu, mu
vloxseg2ei32.v  v8, (a0), v16
vsetvli zero, zero, e32, m4, tu, mu
vloxseg2ei32.v  v8, (a0), v16
vsetvli zero, zero, e64, m1, tu, mu
vloxseg2ei64.v  v8, (a0), v16
vsetvli zero, zero, e64, m2, tu, mu
vloxseg2ei64.v  v8, (a0), v16
vsetvli zero, zero, e64, m4, tu, mu
vloxseg2ei64.v  v8, (a0), v16

vsetvli zero, zero, e8, mf8, tu, mu
vloxseg3ei8.v v8, (a0), v16
vsetvli zero, zero, e8, mf4, tu, mu
vloxseg3ei8.v v8, (a0), v16
vsetvli zero, zero, e8, mf2, tu, mu
vloxseg3ei8.v v8, (a0), v16
vsetvli zero, zero, e8, m1, tu, mu
vloxseg3ei8.v v8, (a0), v16
vsetvli zero, zero, e8, m2, tu, mu
vloxseg3ei8.v v8, (a0), v16
vsetvli zero, zero, e16, mf4, tu, mu
vloxseg3ei16.v  v8, (a0), v16
vsetvli zero, zero, e16, mf2, tu, mu
vloxseg3ei16.v  v8, (a0), v16
vsetvli zero, zero, e16, m1, tu, mu
vloxseg3ei16.v  v8, (a0), v16
vsetvli zero, zero, e16, m2, tu, mu
vloxseg3ei16.v  v8, (a0), v16
vsetvli zero, zero, e32, mf2, tu, mu
vloxseg3ei32.v  v8, (a0), v16
vsetvli zero, zero, e32, m1, tu, mu
vloxseg3ei32.v  v8, (a0), v16
vsetvli zero, zero, e32, m2, tu, mu
vloxseg3ei32.v  v8, (a0), v16
vsetvli zero, zero, e64, m1, tu, mu
vloxseg3ei64.v  v8, (a0), v16
vsetvli zero, zero, e64, m2, tu, mu
vloxseg3ei64.v  v8, (a0), v16

vsetvli zero, zero, e8, mf8, tu, mu
vloxseg4ei8.v v8, (a0), v16
vsetvli zero, zero, e8, mf4, tu, mu
vloxseg4ei8.v v8, (a0), v16
vsetvli zero, zero, e8, mf2, tu, mu
vloxseg4ei8.v v8, (a0), v16
vsetvli zero, zero, e8, m1, tu, mu
vloxseg4ei8.v v8, (a0), v16
vsetvli zero, zero, e8, m2, tu, mu
vloxseg4ei8.v v8, (a0), v16
vsetvli zero, zero, e16, mf4, tu, mu
vloxseg4ei16.v  v8, (a0), v16
vsetvli zero, zero, e16, mf2, tu, mu
vloxseg4ei16.v  v8, (a0), v16
vsetvli zero, zero, e16, m1, tu, mu
vloxseg4ei16.v  v8, (a0), v16
vsetvli zero, zero, e16, m2, tu, mu
vloxseg4ei16.v  v8, (a0), v16
vsetvli zero, zero, e32, mf2, tu, mu
vloxseg4ei32.v  v8, (a0), v16
vsetvli zero, zero, e32, m1, tu, mu
vloxseg4ei32.v  v8, (a0), v16
vsetvli zero, zero, e32, m2, tu, mu
vloxseg4ei32.v  v8, (a0), v16
vsetvli zero, zero, e64, m1, tu, mu
vloxseg4ei64.v  v8, (a0), v16
vsetvli zero, zero, e64, m2, tu, mu
vloxseg4ei64.v  v8, (a0), v16

vsetvli zero, zero, e8, mf8, tu, mu
vloxseg5ei8.v v8, (a0), v16
vsetvli zero, zero, e8, mf4, tu, mu
vloxseg5ei8.v v8, (a0), v16
vsetvli zero, zero, e8, mf2, tu, mu
vloxseg5ei8.v v8, (a0), v16
vsetvli zero, zero, e8, m1, tu, mu
vloxseg5ei8.v v8, (a0), v16
vsetvli zero, zero, e16, mf4, tu, mu
vloxseg5ei16.v  v8, (a0), v16
vsetvli zero, zero, e16, mf2, tu, mu
vloxseg5ei16.v  v8, (a0), v16
vsetvli zero, zero, e16, m1, tu, mu
vloxseg5ei16.v  v8, (a0), v16
vsetvli zero, zero, e32, mf2, tu, mu
vloxseg5ei32.v  v8, (a0), v16
vsetvli zero, zero, e32, m1, tu, mu
vloxseg5ei32.v  v8, (a0), v16
vsetvli zero, zero, e64, m1, tu, mu
vloxseg5ei64.v  v8, (a0), v16

vsetvli zero, zero, e8, mf8, tu, mu
vloxseg6ei8.v v8, (a0), v16
vsetvli zero, zero, e8, mf4, tu, mu
vloxseg6ei8.v v8, (a0), v16
vsetvli zero, zero, e8, mf2, tu, mu
vloxseg6ei8.v v8, (a0), v16
vsetvli zero, zero, e8, m1, tu, mu
vloxseg6ei8.v v8, (a0), v16
vsetvli zero, zero, e16, mf4, tu, mu
vloxseg6ei16.v  v8, (a0), v16
vsetvli zero, zero, e16, mf2, tu, mu
vloxseg6ei16.v  v8, (a0), v16
vsetvli zero, zero, e16, m1, tu, mu
vloxseg6ei16.v  v8, (a0), v16
vsetvli zero, zero, e32, mf2, tu, mu
vloxseg6ei32.v  v8, (a0), v16
vsetvli zero, zero, e32, m1, tu, mu
vloxseg6ei32.v  v8, (a0), v16
vsetvli zero, zero, e64, m1, tu, mu
vloxseg6ei64.v  v8, (a0), v16

vsetvli zero, zero, e8, mf8, tu, mu
vloxseg7ei8.v v8, (a0), v16
vsetvli zero, zero, e8, mf4, tu, mu
vloxseg7ei8.v v8, (a0), v16
vsetvli zero, zero, e8, mf2, tu, mu
vloxseg7ei8.v v8, (a0), v16
vsetvli zero, zero, e8, m1, tu, mu
vloxseg7ei8.v v8, (a0), v16
vsetvli zero, zero, e16, mf4, tu, mu
vloxseg7ei16.v  v8, (a0), v16
vsetvli zero, zero, e16, mf2, tu, mu
vloxseg7ei16.v  v8, (a0), v16
vsetvli zero, zero, e16, m1, tu, mu
vloxseg7ei16.v  v8, (a0), v16
vsetvli zero, zero, e32, mf2, tu, mu
vloxseg7ei32.v  v8, (a0), v16
vsetvli zero, zero, e32, m1, tu, mu
vloxseg7ei32.v  v8, (a0), v16
vsetvli zero, zero, e64, m1, tu, mu
vloxseg7ei64.v  v8, (a0), v16

vsetvli zero, zero, e8, mf8, tu, mu
vloxseg8ei8.v v8, (a0), v16
vsetvli zero, zero, e8, mf4, tu, mu
vloxseg8ei8.v v8, (a0), v16
vsetvli zero, zero, e8, mf2, tu, mu
vloxseg8ei8.v v8, (a0), v16
vsetvli zero, zero, e8, m1, tu, mu
vloxseg8ei8.v v8, (a0), v16
vsetvli zero, zero, e16, mf4, tu, mu
vloxseg8ei16.v  v8, (a0), v16
vsetvli zero, zero, e16, mf2, tu, mu
vloxseg8ei16.v  v8, (a0), v16
vsetvli zero, zero, e16, m1, tu, mu
vloxseg8ei16.v  v8, (a0), v16
vsetvli zero, zero, e32, mf2, tu, mu
vloxseg8ei32.v  v8, (a0), v16
vsetvli zero, zero, e32, m1, tu, mu
vloxseg8ei32.v  v8, (a0), v16
vsetvli zero, zero, e64, m1, tu, mu
vloxseg8ei64.v  v8, (a0), v16

vsetvli zero, zero, e8, mf8, tu, mu
vsuxseg2ei8.v v8, (a0), v16
vsetvli zero, zero, e8, mf4, tu, mu
vsuxseg2ei8.v v8, (a0), v16
vsetvli zero, zero, e8, mf2, tu, mu
vsuxseg2ei8.v v8, (a0), v16
vsetvli zero, zero, e8, m1, tu, mu
vsuxseg2ei8.v v8, (a0), v16
vsetvli zero, zero, e8, m2, tu, mu
vsuxseg2ei8.v v8, (a0), v16
vsetvli zero, zero, e16, mf4, tu, mu
vsuxseg2ei16.v  v8, (a0), v16
vsetvli zero, zero, e16, mf2, tu, mu
vsuxseg2ei16.v  v8, (a0), v16
vsetvli zero, zero, e16, m1, tu, mu
vsuxseg2ei16.v  v8, (a0), v16
vsetvli zero, zero, e16, m2, tu, mu
vsuxseg2ei16.v  v8, (a0), v16
vsetvli zero, zero, e32, mf2, tu, mu
vsuxseg2ei32.v  v8, (a0), v16
vsetvli zero, zero, e32, m1, tu, mu
vsuxseg2ei32.v  v8, (a0), v16
vsetvli zero, zero, e32, m2, tu, mu
vsuxseg2ei32.v  v8, (a0), v16
vsetvli zero, zero, e64, m1, tu, mu
vsuxseg2ei64.v  v8, (a0), v16
vsetvli zero, zero, e64, m2, tu, mu
vsuxseg2ei64.v  v8, (a0), v16

vsetvli zero, zero, e8, mf8, tu, mu
vsuxseg3ei8.v v8, (a0), v16
vsetvli zero, zero, e8, mf4, tu, mu
vsuxseg3ei8.v v8, (a0), v16
vsetvli zero, zero, e8, mf2, tu, mu
vsuxseg3ei8.v v8, (a0), v16
vsetvli zero, zero, e8, m1, tu, mu
vsuxseg3ei8.v v8, (a0), v16
vsetvli zero, zero, e8, m2, tu, mu
vsuxseg3ei8.v v8, (a0), v16
vsetvli zero, zero, e16, mf4, tu, mu
vsuxseg3ei16.v  v8, (a0), v16
vsetvli zero, zero, e16, mf2, tu, mu
vsuxseg3ei16.v  v8, (a0), v16
vsetvli zero, zero, e16, m1, tu, mu
vsuxseg3ei16.v  v8, (a0), v16
vsetvli zero, zero, e16, m2, tu, mu
vsuxseg3ei16.v  v8, (a0), v16
vsetvli zero, zero, e32, mf2, tu, mu
vsuxseg3ei32.v  v8, (a0), v16
vsetvli zero, zero, e32, m1, tu, mu
vsuxseg3ei32.v  v8, (a0), v16
vsetvli zero, zero, e32, m2, tu, mu
vsuxseg3ei32.v  v8, (a0), v16
vsetvli zero, zero, e64, m1, tu, mu
vsuxseg3ei64.v  v8, (a0), v16
vsetvli zero, zero, e64, m2, tu, mu
vsuxseg3ei64.v  v8, (a0), v16

vsetvli zero, zero, e8, mf8, tu, mu
vsuxseg4ei8.v v8, (a0), v16
vsetvli zero, zero, e8, mf4, tu, mu
vsuxseg4ei8.v v8, (a0), v16
vsetvli zero, zero, e8, mf2, tu, mu
vsuxseg4ei8.v v8, (a0), v16
vsetvli zero, zero, e8, m1, tu, mu
vsuxseg4ei8.v v8, (a0), v16
vsetvli zero, zero, e8, m2, tu, mu
vsuxseg4ei8.v v8, (a0), v16
vsetvli zero, zero, e16, mf4, tu, mu
vsuxseg4ei16.v  v8, (a0), v16
vsetvli zero, zero, e16, mf2, tu, mu
vsuxseg4ei16.v  v8, (a0), v16
vsetvli zero, zero, e16, m1, tu, mu
vsuxseg4ei16.v  v8, (a0), v16
vsetvli zero, zero, e16, m2, tu, mu
vsuxseg4ei16.v  v8, (a0), v16
vsetvli zero, zero, e32, mf2, tu, mu
vsuxseg4ei32.v  v8, (a0), v16
vsetvli zero, zero, e32, m1, tu, mu
vsuxseg4ei32.v  v8, (a0), v16
vsetvli zero, zero, e32, m2, tu, mu
vsuxseg4ei32.v  v8, (a0), v16
vsetvli zero, zero, e64, m1, tu, mu
vsuxseg4ei64.v  v8, (a0), v16
vsetvli zero, zero, e64, m2, tu, mu
vsuxseg4ei64.v  v8, (a0), v16

vsetvli zero, zero, e8, mf8, tu, mu
vsuxseg5ei8.v v8, (a0), v16
vsetvli zero, zero, e8, mf4, tu, mu
vsuxseg5ei8.v v8, (a0), v16
vsetvli zero, zero, e8, mf2, tu, mu
vsuxseg5ei8.v v8, (a0), v16
vsetvli zero, zero, e8, m1, tu, mu
vsuxseg5ei8.v v8, (a0), v16
vsetvli zero, zero, e16, mf4, tu, mu
vsuxseg5ei16.v  v8, (a0), v16
vsetvli zero, zero, e16, mf2, tu, mu
vsuxseg5ei16.v  v8, (a0), v16
vsetvli zero, zero, e16, m1, tu, mu
vsuxseg5ei16.v  v8, (a0), v16
vsetvli zero, zero, e32, mf2, tu, mu
vsuxseg5ei32.v  v8, (a0), v16
vsetvli zero, zero, e32, m1, tu, mu
vsuxseg5ei32.v  v8, (a0), v16
vsetvli zero, zero, e64, m1, tu, mu
vsuxseg5ei64.v  v8, (a0), v16

vsetvli zero, zero, e8, mf8, tu, mu
vsuxseg6ei8.v v8, (a0), v16
vsetvli zero, zero, e8, mf4, tu, mu
vsuxseg6ei8.v v8, (a0), v16
vsetvli zero, zero, e8, mf2, tu, mu
vsuxseg6ei8.v v8, (a0), v16
vsetvli zero, zero, e8, m1, tu, mu
vsuxseg6ei8.v v8, (a0), v16
vsetvli zero, zero, e16, mf4, tu, mu
vsuxseg6ei16.v  v8, (a0), v16
vsetvli zero, zero, e16, mf2, tu, mu
vsuxseg6ei16.v  v8, (a0), v16
vsetvli zero, zero, e16, m1, tu, mu
vsuxseg6ei16.v  v8, (a0), v16
vsetvli zero, zero, e32, mf2, tu, mu
vsuxseg6ei32.v  v8, (a0), v16
vsetvli zero, zero, e32, m1, tu, mu
vsuxseg6ei32.v  v8, (a0), v16
vsetvli zero, zero, e64, m1, tu, mu
vsuxseg6ei64.v  v8, (a0), v16

vsetvli zero, zero, e8, mf8, tu, mu
vsuxseg7ei8.v v8, (a0), v16
vsetvli zero, zero, e8, mf4, tu, mu
vsuxseg7ei8.v v8, (a0), v16
vsetvli zero, zero, e8, mf2, tu, mu
vsuxseg7ei8.v v8, (a0), v16
vsetvli zero, zero, e8, m1, tu, mu
vsuxseg7ei8.v v8, (a0), v16
vsetvli zero, zero, e16, mf4, tu, mu
vsuxseg7ei16.v  v8, (a0), v16
vsetvli zero, zero, e16, mf2, tu, mu
vsuxseg7ei16.v  v8, (a0), v16
vsetvli zero, zero, e16, m1, tu, mu
vsuxseg7ei16.v  v8, (a0), v16
vsetvli zero, zero, e32, mf2, tu, mu
vsuxseg7ei32.v  v8, (a0), v16
vsetvli zero, zero, e32, m1, tu, mu
vsuxseg7ei32.v  v8, (a0), v16
vsetvli zero, zero, e64, m1, tu, mu
vsuxseg7ei64.v  v8, (a0), v16

vsetvli zero, zero, e8, mf8, tu, mu
vsuxseg8ei8.v v8, (a0), v16
vsetvli zero, zero, e8, mf4, tu, mu
vsuxseg8ei8.v v8, (a0), v16
vsetvli zero, zero, e8, mf2, tu, mu
vsuxseg8ei8.v v8, (a0), v16
vsetvli zero, zero, e8, m1, tu, mu
vsuxseg8ei8.v v8, (a0), v16
vsetvli zero, zero, e16, mf4, tu, mu
vsuxseg8ei16.v  v8, (a0), v16
vsetvli zero, zero, e16, mf2, tu, mu
vsuxseg8ei16.v  v8, (a0), v16
vsetvli zero, zero, e16, m1, tu, mu
vsuxseg8ei16.v  v8, (a0), v16
vsetvli zero, zero, e32, mf2, tu, mu
vsuxseg8ei32.v  v8, (a0), v16
vsetvli zero, zero, e32, m1, tu, mu
vsuxseg8ei32.v  v8, (a0), v16
vsetvli zero, zero, e64, m1, tu, mu
vsuxseg8ei64.v  v8, (a0), v16

vsetvli zero, zero, e8, mf8, tu, mu
vsoxseg2ei8.v v8, (a0), v16
vsetvli zero, zero, e8, mf4, tu, mu
vsoxseg2ei8.v v8, (a0), v16
vsetvli zero, zero, e8, mf2, tu, mu
vsoxseg2ei8.v v8, (a0), v16
vsetvli zero, zero, e8, m1, tu, mu
vsoxseg2ei8.v v8, (a0), v16
vsetvli zero, zero, e8, m2, tu, mu
vsoxseg2ei8.v v8, (a0), v16
vsetvli zero, zero, e8, m4, tu, mu
vsoxseg2ei8.v v8, (a0), v16
vsetvli zero, zero, e16, mf4, tu, mu
vsoxseg2ei16.v  v8, (a0), v16
vsetvli zero, zero, e16, mf2, tu, mu
vsoxseg2ei16.v  v8, (a0), v16
vsetvli zero, zero, e16, m1, tu, mu
vsoxseg2ei16.v  v8, (a0), v16
vsetvli zero, zero, e16, m2, tu, mu
vsoxseg2ei16.v  v8, (a0), v16
vsetvli zero, zero, e16, m4, tu, mu
vsoxseg2ei16.v  v8, (a0), v16
vsetvli zero, zero, e32, mf2, tu, mu
vsoxseg2ei32.v  v8, (a0), v16
vsetvli zero, zero, e32, m1, tu, mu
vsoxseg2ei32.v  v8, (a0), v16
vsetvli zero, zero, e32, m2, tu, mu
vsoxseg2ei32.v  v8, (a0), v16
vsetvli zero, zero, e32, m4, tu, mu
vsoxseg2ei32.v  v8, (a0), v16
vsetvli zero, zero, e64, m1, tu, mu
vsoxseg2ei64.v  v8, (a0), v16
vsetvli zero, zero, e64, m2, tu, mu
vsoxseg2ei64.v  v8, (a0), v16
vsetvli zero, zero, e64, m4, tu, mu
vsoxseg2ei64.v  v8, (a0), v16

vsetvli zero, zero, e8, mf8, tu, mu
vsoxseg3ei8.v v8, (a0), v16
vsetvli zero, zero, e8, mf4, tu, mu
vsoxseg3ei8.v v8, (a0), v16
vsetvli zero, zero, e8, mf2, tu, mu
vsoxseg3ei8.v v8, (a0), v16
vsetvli zero, zero, e8, m1, tu, mu
vsoxseg3ei8.v v8, (a0), v16
vsetvli zero, zero, e8, m2, tu, mu
vsoxseg3ei8.v v8, (a0), v16
vsetvli zero, zero, e16, mf4, tu, mu
vsoxseg3ei16.v  v8, (a0), v16
vsetvli zero, zero, e16, mf2, tu, mu
vsoxseg3ei16.v  v8, (a0), v16
vsetvli zero, zero, e16, m1, tu, mu
vsoxseg3ei16.v  v8, (a0), v16
vsetvli zero, zero, e16, m2, tu, mu
vsoxseg3ei16.v  v8, (a0), v16
vsetvli zero, zero, e32, mf2, tu, mu
vsoxseg3ei32.v  v8, (a0), v16
vsetvli zero, zero, e32, m1, tu, mu
vsoxseg3ei32.v  v8, (a0), v16
vsetvli zero, zero, e32, m2, tu, mu
vsoxseg3ei32.v  v8, (a0), v16
vsetvli zero, zero, e64, m1, tu, mu
vsoxseg3ei64.v  v8, (a0), v16
vsetvli zero, zero, e64, m2, tu, mu
vsoxseg3ei64.v  v8, (a0), v16

vsetvli zero, zero, e8, mf8, tu, mu
vsoxseg4ei8.v v8, (a0), v16
vsetvli zero, zero, e8, mf4, tu, mu
vsoxseg4ei8.v v8, (a0), v16
vsetvli zero, zero, e8, mf2, tu, mu
vsoxseg4ei8.v v8, (a0), v16
vsetvli zero, zero, e8, m1, tu, mu
vsoxseg4ei8.v v8, (a0), v16
vsetvli zero, zero, e8, m2, tu, mu
vsoxseg4ei8.v v8, (a0), v16
vsetvli zero, zero, e16, mf4, tu, mu
vsoxseg4ei16.v  v8, (a0), v16
vsetvli zero, zero, e16, mf2, tu, mu
vsoxseg4ei16.v  v8, (a0), v16
vsetvli zero, zero, e16, m1, tu, mu
vsoxseg4ei16.v  v8, (a0), v16
vsetvli zero, zero, e16, m2, tu, mu
vsoxseg4ei16.v  v8, (a0), v16
vsetvli zero, zero, e32, mf2, tu, mu
vsoxseg4ei32.v  v8, (a0), v16
vsetvli zero, zero, e32, m1, tu, mu
vsoxseg4ei32.v  v8, (a0), v16
vsetvli zero, zero, e32, m2, tu, mu
vsoxseg4ei32.v  v8, (a0), v16
vsetvli zero, zero, e64, m1, tu, mu
vsoxseg4ei64.v  v8, (a0), v16
vsetvli zero, zero, e64, m2, tu, mu
vsoxseg4ei64.v  v8, (a0), v16

vsetvli zero, zero, e8, mf8, tu, mu
vsoxseg5ei8.v v8, (a0), v16
vsetvli zero, zero, e8, mf4, tu, mu
vsoxseg5ei8.v v8, (a0), v16
vsetvli zero, zero, e8, mf2, tu, mu
vsoxseg5ei8.v v8, (a0), v16
vsetvli zero, zero, e8, m1, tu, mu
vsoxseg5ei8.v v8, (a0), v16
vsetvli zero, zero, e16, mf4, tu, mu
vsoxseg5ei16.v  v8, (a0), v16
vsetvli zero, zero, e16, mf2, tu, mu
vsoxseg5ei16.v  v8, (a0), v16
vsetvli zero, zero, e16, m1, tu, mu
vsoxseg5ei16.v  v8, (a0), v16
vsetvli zero, zero, e32, mf2, tu, mu
vsoxseg5ei32.v  v8, (a0), v16
vsetvli zero, zero, e32, m1, tu, mu
vsoxseg5ei32.v  v8, (a0), v16
vsetvli zero, zero, e64, m1, tu, mu
vsoxseg5ei64.v  v8, (a0), v16

vsetvli zero, zero, e8, mf8, tu, mu
vsoxseg6ei8.v v8, (a0), v16
vsetvli zero, zero, e8, mf4, tu, mu
vsoxseg6ei8.v v8, (a0), v16
vsetvli zero, zero, e8, mf2, tu, mu
vsoxseg6ei8.v v8, (a0), v16
vsetvli zero, zero, e8, m1, tu, mu
vsoxseg6ei8.v v8, (a0), v16
vsetvli zero, zero, e16, mf4, tu, mu
vsoxseg6ei16.v  v8, (a0), v16
vsetvli zero, zero, e16, mf2, tu, mu
vsoxseg6ei16.v  v8, (a0), v16
vsetvli zero, zero, e16, m1, tu, mu
vsoxseg6ei16.v  v8, (a0), v16
vsetvli zero, zero, e32, mf2, tu, mu
vsoxseg6ei32.v  v8, (a0), v16
vsetvli zero, zero, e32, m1, tu, mu
vsoxseg6ei32.v  v8, (a0), v16
vsetvli zero, zero, e64, m1, tu, mu
vsoxseg6ei64.v  v8, (a0), v16

vsetvli zero, zero, e8, mf8, tu, mu
vsoxseg7ei8.v v8, (a0), v16
vsetvli zero, zero, e8, mf4, tu, mu
vsoxseg7ei8.v v8, (a0), v16
vsetvli zero, zero, e8, mf2, tu, mu
vsoxseg7ei8.v v8, (a0), v16
vsetvli zero, zero, e8, m1, tu, mu
vsoxseg7ei8.v v8, (a0), v16
vsetvli zero, zero, e16, mf4, tu, mu
vsoxseg7ei16.v  v8, (a0), v16
vsetvli zero, zero, e16, mf2, tu, mu
vsoxseg7ei16.v  v8, (a0), v16
vsetvli zero, zero, e16, m1, tu, mu
vsoxseg7ei16.v  v8, (a0), v16
vsetvli zero, zero, e32, mf2, tu, mu
vsoxseg7ei32.v  v8, (a0), v16
vsetvli zero, zero, e32, m1, tu, mu
vsoxseg7ei32.v  v8, (a0), v16
vsetvli zero, zero, e64, m1, tu, mu
vsoxseg7ei64.v  v8, (a0), v16

vsetvli zero, zero, e8, mf8, tu, mu
vsoxseg8ei8.v v8, (a0), v16
vsetvli zero, zero, e8, mf4, tu, mu
vsoxseg8ei8.v v8, (a0), v16
vsetvli zero, zero, e8, mf2, tu, mu
vsoxseg8ei8.v v8, (a0), v16
vsetvli zero, zero, e8, m1, tu, mu
vsoxseg8ei8.v v8, (a0), v16
vsetvli zero, zero, e16, mf4, tu, mu
vsoxseg8ei16.v  v8, (a0), v16
vsetvli zero, zero, e16, mf2, tu, mu
vsoxseg8ei16.v  v8, (a0), v16
vsetvli zero, zero, e16, m1, tu, mu
vsoxseg8ei16.v  v8, (a0), v16
vsetvli zero, zero, e32, mf2, tu, mu
vsoxseg8ei32.v  v8, (a0), v16
vsetvli zero, zero, e32, m1, tu, mu
vsoxseg8ei32.v  v8, (a0), v16
vsetvli zero, zero, e64, m1, tu, mu
vsoxseg8ei64.v  v8, (a0), v16
