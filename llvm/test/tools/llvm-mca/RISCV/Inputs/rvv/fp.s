# Floating point operations

vsetvli x28, x0, e16, mf2, tu, mu
vmfeq.vf v8, v8, ft0
vsetvli x28, x0, e16, mf4, tu, mu
vmfeq.vf v8, v8, ft0
vsetvli x28, x0, e16, m1, tu, mu
vmfeq.vf v8, v8, ft0
vsetvli x28, x0, e16, m2, tu, mu
vmfeq.vf v8, v8, ft0
vsetvli x28, x0, e16, m4, tu, mu
vmfeq.vf v8, v8, ft0
vsetvli x28, x0, e16, m8, tu, mu
vmfeq.vf v8, v8, ft0
vsetvli x28, x0, e32, mf2, tu, mu
vmfeq.vf v8, v8, ft0
vsetvli x28, x0, e32, m1, tu, mu
vmfeq.vf v8, v8, ft0
vsetvli x28, x0, e32, m2, tu, mu
vmfeq.vf v8, v8, ft0
vsetvli x28, x0, e32, m4, tu, mu
vmfeq.vf v8, v8, ft0
vsetvli x28, x0, e32, m8, tu, mu
vmfeq.vf v8, v8, ft0
vsetvli x28, x0, e64, m1, tu, mu
vmfeq.vf v8, v8, ft0
vsetvli x28, x0, e64, m2, tu, mu
vmfeq.vf v8, v8, ft0
vsetvli x28, x0, e64, m4, tu, mu
vmfeq.vf v8, v8, ft0
vsetvli x28, x0, e64, m8, tu, mu
vmfeq.vf v8, v8, ft0

vsetvli x28, x0, e16, mf2, tu, mu
vmfeq.vv v8, v8, v8
vsetvli x28, x0, e16, mf4, tu, mu
vmfeq.vv v8, v8, v8
vsetvli x28, x0, e16, m1, tu, mu
vmfeq.vv v8, v8, v8
vsetvli x28, x0, e16, m2, tu, mu
vmfeq.vv v8, v8, v8
vsetvli x28, x0, e16, m4, tu, mu
vmfeq.vv v8, v8, v8
vsetvli x28, x0, e16, m8, tu, mu
vmfeq.vv v8, v8, v8
vsetvli x28, x0, e32, mf2, tu, mu
vmfeq.vv v8, v8, v8
vsetvli x28, x0, e32, m1, tu, mu
vmfeq.vv v8, v8, v8
vsetvli x28, x0, e32, m2, tu, mu
vmfeq.vv v8, v8, v8
vsetvli x28, x0, e32, m4, tu, mu
vmfeq.vv v8, v8, v8
vsetvli x28, x0, e32, m8, tu, mu
vmfeq.vv v8, v8, v8
vsetvli x28, x0, e64, m1, tu, mu
vmfeq.vv v8, v8, v8
vsetvli x28, x0, e64, m2, tu, mu
vmfeq.vv v8, v8, v8
vsetvli x28, x0, e64, m4, tu, mu
vmfeq.vv v8, v8, v8
vsetvli x28, x0, e64, m8, tu, mu
vmfeq.vv v8, v8, v8

vsetvli x28, x0, e16, mf2, tu, mu
vmfge.vf v8, v8, ft0
vsetvli x28, x0, e16, mf4, tu, mu
vmfge.vf v8, v8, ft0
vsetvli x28, x0, e16, m1, tu, mu
vmfge.vf v8, v8, ft0
vsetvli x28, x0, e16, m2, tu, mu
vmfge.vf v8, v8, ft0
vsetvli x28, x0, e16, m4, tu, mu
vmfge.vf v8, v8, ft0
vsetvli x28, x0, e16, m8, tu, mu
vmfge.vf v8, v8, ft0
vsetvli x28, x0, e32, mf2, tu, mu
vmfge.vf v8, v8, ft0
vsetvli x28, x0, e32, m1, tu, mu
vmfge.vf v8, v8, ft0
vsetvli x28, x0, e32, m2, tu, mu
vmfge.vf v8, v8, ft0
vsetvli x28, x0, e32, m4, tu, mu
vmfge.vf v8, v8, ft0
vsetvli x28, x0, e32, m8, tu, mu
vmfge.vf v8, v8, ft0
vsetvli x28, x0, e64, m1, tu, mu
vmfge.vf v8, v8, ft0
vsetvli x28, x0, e64, m2, tu, mu
vmfge.vf v8, v8, ft0
vsetvli x28, x0, e64, m4, tu, mu
vmfge.vf v8, v8, ft0
vsetvli x28, x0, e64, m8, tu, mu
vmfge.vf v8, v8, ft0

vsetvli x28, x0, e16, mf2, tu, mu
vmfge.vv v8, v8, v8
vsetvli x28, x0, e16, mf4, tu, mu
vmfge.vv v8, v8, v8
vsetvli x28, x0, e16, m1, tu, mu
vmfge.vv v8, v8, v8
vsetvli x28, x0, e16, m2, tu, mu
vmfge.vv v8, v8, v8
vsetvli x28, x0, e16, m4, tu, mu
vmfge.vv v8, v8, v8
vsetvli x28, x0, e16, m8, tu, mu
vmfge.vv v8, v8, v8
vsetvli x28, x0, e32, mf2, tu, mu
vmfge.vv v8, v8, v8
vsetvli x28, x0, e32, m1, tu, mu
vmfge.vv v8, v8, v8
vsetvli x28, x0, e32, m2, tu, mu
vmfge.vv v8, v8, v8
vsetvli x28, x0, e32, m4, tu, mu
vmfge.vv v8, v8, v8
vsetvli x28, x0, e32, m8, tu, mu
vmfge.vv v8, v8, v8
vsetvli x28, x0, e64, m1, tu, mu
vmfge.vv v8, v8, v8
vsetvli x28, x0, e64, m2, tu, mu
vmfge.vv v8, v8, v8
vsetvli x28, x0, e64, m4, tu, mu
vmfge.vv v8, v8, v8
vsetvli x28, x0, e64, m8, tu, mu
vmfge.vv v8, v8, v8

vsetvli x28, x0, e16, mf2, tu, mu
vmfgt.vf v8, v8, ft0
vsetvli x28, x0, e16, mf4, tu, mu
vmfgt.vf v8, v8, ft0
vsetvli x28, x0, e16, m1, tu, mu
vmfgt.vf v8, v8, ft0
vsetvli x28, x0, e16, m2, tu, mu
vmfgt.vf v8, v8, ft0
vsetvli x28, x0, e16, m4, tu, mu
vmfgt.vf v8, v8, ft0
vsetvli x28, x0, e16, m8, tu, mu
vmfgt.vf v8, v8, ft0
vsetvli x28, x0, e32, mf2, tu, mu
vmfgt.vf v8, v8, ft0
vsetvli x28, x0, e32, m1, tu, mu
vmfgt.vf v8, v8, ft0
vsetvli x28, x0, e32, m2, tu, mu
vmfgt.vf v8, v8, ft0
vsetvli x28, x0, e32, m4, tu, mu
vmfgt.vf v8, v8, ft0
vsetvli x28, x0, e32, m8, tu, mu
vmfgt.vf v8, v8, ft0
vsetvli x28, x0, e64, m1, tu, mu
vmfgt.vf v8, v8, ft0
vsetvli x28, x0, e64, m2, tu, mu
vmfgt.vf v8, v8, ft0
vsetvli x28, x0, e64, m4, tu, mu
vmfgt.vf v8, v8, ft0
vsetvli x28, x0, e64, m8, tu, mu
vmfgt.vf v8, v8, ft0

vsetvli x28, x0, e16, mf2, tu, mu
vmfgt.vv v8, v8, v8
vsetvli x28, x0, e16, mf4, tu, mu
vmfgt.vv v8, v8, v8
vsetvli x28, x0, e16, m1, tu, mu
vmfgt.vv v8, v8, v8
vsetvli x28, x0, e16, m2, tu, mu
vmfgt.vv v8, v8, v8
vsetvli x28, x0, e16, m4, tu, mu
vmfgt.vv v8, v8, v8
vsetvli x28, x0, e16, m8, tu, mu
vmfgt.vv v8, v8, v8
vsetvli x28, x0, e32, mf2, tu, mu
vmfgt.vv v8, v8, v8
vsetvli x28, x0, e32, m1, tu, mu
vmfgt.vv v8, v8, v8
vsetvli x28, x0, e32, m2, tu, mu
vmfgt.vv v8, v8, v8
vsetvli x28, x0, e32, m4, tu, mu
vmfgt.vv v8, v8, v8
vsetvli x28, x0, e32, m8, tu, mu
vmfgt.vv v8, v8, v8
vsetvli x28, x0, e64, m1, tu, mu
vmfgt.vv v8, v8, v8
vsetvli x28, x0, e64, m2, tu, mu
vmfgt.vv v8, v8, v8
vsetvli x28, x0, e64, m4, tu, mu
vmfgt.vv v8, v8, v8
vsetvli x28, x0, e64, m8, tu, mu
vmfgt.vv v8, v8, v8

vsetvli x28, x0, e16, mf2, tu, mu
vmfle.vf v8, v8, ft0
vsetvli x28, x0, e16, mf4, tu, mu
vmfle.vf v8, v8, ft0
vsetvli x28, x0, e16, m1, tu, mu
vmfle.vf v8, v8, ft0
vsetvli x28, x0, e16, m2, tu, mu
vmfle.vf v8, v8, ft0
vsetvli x28, x0, e16, m4, tu, mu
vmfle.vf v8, v8, ft0
vsetvli x28, x0, e16, m8, tu, mu
vmfle.vf v8, v8, ft0
vsetvli x28, x0, e32, mf2, tu, mu
vmfle.vf v8, v8, ft0
vsetvli x28, x0, e32, m1, tu, mu
vmfle.vf v8, v8, ft0
vsetvli x28, x0, e32, m2, tu, mu
vmfle.vf v8, v8, ft0
vsetvli x28, x0, e32, m4, tu, mu
vmfle.vf v8, v8, ft0
vsetvli x28, x0, e32, m8, tu, mu
vmfle.vf v8, v8, ft0
vsetvli x28, x0, e64, m1, tu, mu
vmfle.vf v8, v8, ft0
vsetvli x28, x0, e64, m2, tu, mu
vmfle.vf v8, v8, ft0
vsetvli x28, x0, e64, m4, tu, mu
vmfle.vf v8, v8, ft0
vsetvli x28, x0, e64, m8, tu, mu
vmfle.vf v8, v8, ft0

vsetvli x28, x0, e16, mf2, tu, mu
vmfle.vv v8, v8, v8
vsetvli x28, x0, e16, mf4, tu, mu
vmfle.vv v8, v8, v8
vsetvli x28, x0, e16, m1, tu, mu
vmfle.vv v8, v8, v8
vsetvli x28, x0, e16, m2, tu, mu
vmfle.vv v8, v8, v8
vsetvli x28, x0, e16, m4, tu, mu
vmfle.vv v8, v8, v8
vsetvli x28, x0, e16, m8, tu, mu
vmfle.vv v8, v8, v8
vsetvli x28, x0, e32, mf2, tu, mu
vmfle.vv v8, v8, v8
vsetvli x28, x0, e32, m1, tu, mu
vmfle.vv v8, v8, v8
vsetvli x28, x0, e32, m2, tu, mu
vmfle.vv v8, v8, v8
vsetvli x28, x0, e32, m4, tu, mu
vmfle.vv v8, v8, v8
vsetvli x28, x0, e32, m8, tu, mu
vmfle.vv v8, v8, v8
vsetvli x28, x0, e64, m1, tu, mu
vmfle.vv v8, v8, v8
vsetvli x28, x0, e64, m2, tu, mu
vmfle.vv v8, v8, v8
vsetvli x28, x0, e64, m4, tu, mu
vmfle.vv v8, v8, v8
vsetvli x28, x0, e64, m8, tu, mu
vmfle.vv v8, v8, v8

vsetvli x28, x0, e16, mf2, tu, mu
vmflt.vf v8, v8, ft0
vsetvli x28, x0, e16, mf4, tu, mu
vmflt.vf v8, v8, ft0
vsetvli x28, x0, e16, m1, tu, mu
vmflt.vf v8, v8, ft0
vsetvli x28, x0, e16, m2, tu, mu
vmflt.vf v8, v8, ft0
vsetvli x28, x0, e16, m4, tu, mu
vmflt.vf v8, v8, ft0
vsetvli x28, x0, e16, m8, tu, mu
vmflt.vf v8, v8, ft0
vsetvli x28, x0, e32, mf2, tu, mu
vmflt.vf v8, v8, ft0
vsetvli x28, x0, e32, m1, tu, mu
vmflt.vf v8, v8, ft0
vsetvli x28, x0, e32, m2, tu, mu
vmflt.vf v8, v8, ft0
vsetvli x28, x0, e32, m4, tu, mu
vmflt.vf v8, v8, ft0
vsetvli x28, x0, e32, m8, tu, mu
vmflt.vf v8, v8, ft0
vsetvli x28, x0, e64, m1, tu, mu
vmflt.vf v8, v8, ft0
vsetvli x28, x0, e64, m2, tu, mu
vmflt.vf v8, v8, ft0
vsetvli x28, x0, e64, m4, tu, mu
vmflt.vf v8, v8, ft0
vsetvli x28, x0, e64, m8, tu, mu
vmflt.vf v8, v8, ft0

vsetvli x28, x0, e16, mf2, tu, mu
vmflt.vv v8, v8, v8
vsetvli x28, x0, e16, mf4, tu, mu
vmflt.vv v8, v8, v8
vsetvli x28, x0, e16, m1, tu, mu
vmflt.vv v8, v8, v8
vsetvli x28, x0, e16, m2, tu, mu
vmflt.vv v8, v8, v8
vsetvli x28, x0, e16, m4, tu, mu
vmflt.vv v8, v8, v8
vsetvli x28, x0, e16, m8, tu, mu
vmflt.vv v8, v8, v8
vsetvli x28, x0, e32, mf2, tu, mu
vmflt.vv v8, v8, v8
vsetvli x28, x0, e32, m1, tu, mu
vmflt.vv v8, v8, v8
vsetvli x28, x0, e32, m2, tu, mu
vmflt.vv v8, v8, v8
vsetvli x28, x0, e32, m4, tu, mu
vmflt.vv v8, v8, v8
vsetvli x28, x0, e32, m8, tu, mu
vmflt.vv v8, v8, v8
vsetvli x28, x0, e64, m1, tu, mu
vmflt.vv v8, v8, v8
vsetvli x28, x0, e64, m2, tu, mu
vmflt.vv v8, v8, v8
vsetvli x28, x0, e64, m4, tu, mu
vmflt.vv v8, v8, v8
vsetvli x28, x0, e64, m8, tu, mu
vmflt.vv v8, v8, v8

vsetvli x28, x0, e16, mf2, tu, mu
vmfne.vf v8, v8, ft0
vsetvli x28, x0, e16, mf4, tu, mu
vmfne.vf v8, v8, ft0
vsetvli x28, x0, e16, m1, tu, mu
vmfne.vf v8, v8, ft0
vsetvli x28, x0, e16, m2, tu, mu
vmfne.vf v8, v8, ft0
vsetvli x28, x0, e16, m4, tu, mu
vmfne.vf v8, v8, ft0
vsetvli x28, x0, e16, m8, tu, mu
vmfne.vf v8, v8, ft0
vsetvli x28, x0, e32, mf2, tu, mu
vmfne.vf v8, v8, ft0
vsetvli x28, x0, e32, m1, tu, mu
vmfne.vf v8, v8, ft0
vsetvli x28, x0, e32, m2, tu, mu
vmfne.vf v8, v8, ft0
vsetvli x28, x0, e32, m4, tu, mu
vmfne.vf v8, v8, ft0
vsetvli x28, x0, e32, m8, tu, mu
vmfne.vf v8, v8, ft0
vsetvli x28, x0, e64, m1, tu, mu
vmfne.vf v8, v8, ft0
vsetvli x28, x0, e64, m2, tu, mu
vmfne.vf v8, v8, ft0
vsetvli x28, x0, e64, m4, tu, mu
vmfne.vf v8, v8, ft0
vsetvli x28, x0, e64, m8, tu, mu
vmfne.vf v8, v8, ft0

vsetvli x28, x0, e16, mf2, tu, mu
vmfne.vv v8, v8, v8
vsetvli x28, x0, e16, mf4, tu, mu
vmfne.vv v8, v8, v8
vsetvli x28, x0, e16, m1, tu, mu
vmfne.vv v8, v8, v8
vsetvli x28, x0, e16, m2, tu, mu
vmfne.vv v8, v8, v8
vsetvli x28, x0, e16, m4, tu, mu
vmfne.vv v8, v8, v8
vsetvli x28, x0, e16, m8, tu, mu
vmfne.vv v8, v8, v8
vsetvli x28, x0, e32, mf2, tu, mu
vmfne.vv v8, v8, v8
vsetvli x28, x0, e32, m1, tu, mu
vmfne.vv v8, v8, v8
vsetvli x28, x0, e32, m2, tu, mu
vmfne.vv v8, v8, v8
vsetvli x28, x0, e32, m4, tu, mu
vmfne.vv v8, v8, v8
vsetvli x28, x0, e32, m8, tu, mu
vmfne.vv v8, v8, v8
vsetvli x28, x0, e64, m1, tu, mu
vmfne.vv v8, v8, v8
vsetvli x28, x0, e64, m2, tu, mu
vmfne.vv v8, v8, v8
vsetvli x28, x0, e64, m4, tu, mu
vmfne.vv v8, v8, v8
vsetvli x28, x0, e64, m8, tu, mu
vmfne.vv v8, v8, v8

vsetvli x28, x0, e16, mf2, tu, mu
vfadd.vv v8, v8, v8
vsetvli x28, x0, e16, mf4, tu, mu
vfadd.vv v8, v8, v8
vsetvli x28, x0, e16, m1, tu, mu
vfadd.vv v8, v8, v8
vsetvli x28, x0, e16, m2, tu, mu
vfadd.vv v8, v8, v8
vsetvli x28, x0, e16, m4, tu, mu
vfadd.vv v8, v8, v8
vsetvli x28, x0, e16, m8, tu, mu
vfadd.vv v8, v8, v8
vsetvli x28, x0, e32, mf2, tu, mu
vfadd.vv v8, v8, v8
vsetvli x28, x0, e32, m1, tu, mu
vfadd.vv v8, v8, v8
vsetvli x28, x0, e32, m2, tu, mu
vfadd.vv v8, v8, v8
vsetvli x28, x0, e32, m4, tu, mu
vfadd.vv v8, v8, v8
vsetvli x28, x0, e32, m8, tu, mu
vfadd.vv v8, v8, v8
vsetvli x28, x0, e64, m1, tu, mu
vfadd.vv v8, v8, v8
vsetvli x28, x0, e64, m2, tu, mu
vfadd.vv v8, v8, v8
vsetvli x28, x0, e64, m4, tu, mu
vfadd.vv v8, v8, v8
vsetvli x28, x0, e64, m8, tu, mu
vfadd.vv v8, v8, v8

vsetvli x28, x0, e16, mf2, tu, mu
vfadd.vf v8, v8, ft0
vsetvli x28, x0, e16, mf4, tu, mu
vfadd.vf v8, v8, ft0
vsetvli x28, x0, e16, m1, tu, mu
vfadd.vf v8, v8, ft0
vsetvli x28, x0, e16, m2, tu, mu
vfadd.vf v8, v8, ft0
vsetvli x28, x0, e16, m4, tu, mu
vfadd.vf v8, v8, ft0
vsetvli x28, x0, e16, m8, tu, mu
vfadd.vf v8, v8, ft0
vsetvli x28, x0, e32, mf2, tu, mu
vfadd.vf v8, v8, ft0
vsetvli x28, x0, e32, m1, tu, mu
vfadd.vf v8, v8, ft0
vsetvli x28, x0, e32, m2, tu, mu
vfadd.vf v8, v8, ft0
vsetvli x28, x0, e32, m4, tu, mu
vfadd.vf v8, v8, ft0
vsetvli x28, x0, e32, m8, tu, mu
vfadd.vf v8, v8, ft0
vsetvli x28, x0, e64, m1, tu, mu
vfadd.vf v8, v8, ft0
vsetvli x28, x0, e64, m2, tu, mu
vfadd.vf v8, v8, ft0
vsetvli x28, x0, e64, m4, tu, mu
vfadd.vf v8, v8, ft0
vsetvli x28, x0, e64, m8, tu, mu
vfadd.vf v8, v8, ft0

vsetvli x28, x0, e16, mf2, tu, mu
vfsub.vv v8, v8, v8
vsetvli x28, x0, e16, mf4, tu, mu
vfsub.vv v8, v8, v8
vsetvli x28, x0, e16, m1, tu, mu
vfsub.vv v8, v8, v8
vsetvli x28, x0, e16, m2, tu, mu
vfsub.vv v8, v8, v8
vsetvli x28, x0, e16, m4, tu, mu
vfsub.vv v8, v8, v8
vsetvli x28, x0, e16, m8, tu, mu
vfsub.vv v8, v8, v8
vsetvli x28, x0, e32, mf2, tu, mu
vfsub.vv v8, v8, v8
vsetvli x28, x0, e32, m1, tu, mu
vfsub.vv v8, v8, v8
vsetvli x28, x0, e32, m2, tu, mu
vfsub.vv v8, v8, v8
vsetvli x28, x0, e32, m4, tu, mu
vfsub.vv v8, v8, v8
vsetvli x28, x0, e32, m8, tu, mu
vfsub.vv v8, v8, v8
vsetvli x28, x0, e64, m1, tu, mu
vfsub.vv v8, v8, v8
vsetvli x28, x0, e64, m2, tu, mu
vfsub.vv v8, v8, v8
vsetvli x28, x0, e64, m4, tu, mu
vfsub.vv v8, v8, v8
vsetvli x28, x0, e64, m8, tu, mu
vfsub.vv v8, v8, v8

vsetvli x28, x0, e16, mf2, tu, mu
vfsub.vf v8, v8, ft0
vsetvli x28, x0, e16, mf4, tu, mu
vfsub.vf v8, v8, ft0
vsetvli x28, x0, e16, m1, tu, mu
vfsub.vf v8, v8, ft0
vsetvli x28, x0, e16, m2, tu, mu
vfsub.vf v8, v8, ft0
vsetvli x28, x0, e16, m4, tu, mu
vfsub.vf v8, v8, ft0
vsetvli x28, x0, e16, m8, tu, mu
vfsub.vf v8, v8, ft0
vsetvli x28, x0, e32, mf2, tu, mu
vfsub.vf v8, v8, ft0
vsetvli x28, x0, e32, m1, tu, mu
vfsub.vf v8, v8, ft0
vsetvli x28, x0, e32, m2, tu, mu
vfsub.vf v8, v8, ft0
vsetvli x28, x0, e32, m4, tu, mu
vfsub.vf v8, v8, ft0
vsetvli x28, x0, e32, m8, tu, mu
vfsub.vf v8, v8, ft0
vsetvli x28, x0, e64, m1, tu, mu
vfsub.vf v8, v8, ft0
vsetvli x28, x0, e64, m2, tu, mu
vfsub.vf v8, v8, ft0
vsetvli x28, x0, e64, m4, tu, mu
vfsub.vf v8, v8, ft0
vsetvli x28, x0, e64, m8, tu, mu
vfsub.vf v8, v8, ft0

vsetvli x28, x0, e16, mf2, tu, mu
vfclass.v v8, v8
vsetvli x28, x0, e16, mf4, tu, mu
vfclass.v v8, v8
vsetvli x28, x0, e16, m1, tu, mu
vfclass.v v8, v8
vsetvli x28, x0, e16, m2, tu, mu
vfclass.v v8, v8
vsetvli x28, x0, e16, m4, tu, mu
vfclass.v v8, v8
vsetvli x28, x0, e16, m8, tu, mu
vfclass.v v8, v8
vsetvli x28, x0, e32, mf2, tu, mu
vfclass.v v8, v8
vsetvli x28, x0, e32, m1, tu, mu
vfclass.v v8, v8
vsetvli x28, x0, e32, m2, tu, mu
vfclass.v v8, v8
vsetvli x28, x0, e32, m4, tu, mu
vfclass.v v8, v8
vsetvli x28, x0, e32, m8, tu, mu
vfclass.v v8, v8
vsetvli x28, x0, e64, m1, tu, mu
vfclass.v v8, v8
vsetvli x28, x0, e64, m2, tu, mu
vfclass.v v8, v8
vsetvli x28, x0, e64, m4, tu, mu
vfclass.v v8, v8
vsetvli x28, x0, e64, m8, tu, mu
vfclass.v v8, v8

vsetvli x28, x0, e16, mf2, tu, mu
vfdiv.vv v8, v8, v8
vsetvli x28, x0, e16, mf4, tu, mu
vfdiv.vv v8, v8, v8
vsetvli x28, x0, e16, m1, tu, mu
vfdiv.vv v8, v8, v8
vsetvli x28, x0, e16, m2, tu, mu
vfdiv.vv v8, v8, v8
vsetvli x28, x0, e16, m4, tu, mu
vfdiv.vv v8, v8, v8
vsetvli x28, x0, e16, m8, tu, mu
vfdiv.vv v8, v8, v8
vsetvli x28, x0, e32, mf2, tu, mu
vfdiv.vv v8, v8, v8
vsetvli x28, x0, e32, m1, tu, mu
vfdiv.vv v8, v8, v8
vsetvli x28, x0, e32, m2, tu, mu
vfdiv.vv v8, v8, v8
vsetvli x28, x0, e32, m4, tu, mu
vfdiv.vv v8, v8, v8
vsetvli x28, x0, e32, m8, tu, mu
vfdiv.vv v8, v8, v8
vsetvli x28, x0, e64, m1, tu, mu
vfdiv.vv v8, v8, v8
vsetvli x28, x0, e64, m2, tu, mu
vfdiv.vv v8, v8, v8
vsetvli x28, x0, e64, m4, tu, mu
vfdiv.vv v8, v8, v8
vsetvli x28, x0, e64, m8, tu, mu
vfdiv.vv v8, v8, v8

vsetvli x28, x0, e16, mf2, tu, mu
vfdiv.vf v8, v8, ft0
vsetvli x28, x0, e16, mf4, tu, mu
vfdiv.vf v8, v8, ft0
vsetvli x28, x0, e16, m1, tu, mu
vfdiv.vf v8, v8, ft0
vsetvli x28, x0, e16, m2, tu, mu
vfdiv.vf v8, v8, ft0
vsetvli x28, x0, e16, m4, tu, mu
vfdiv.vf v8, v8, ft0
vsetvli x28, x0, e16, m8, tu, mu
vfdiv.vf v8, v8, ft0
vsetvli x28, x0, e32, mf2, tu, mu
vfdiv.vf v8, v8, ft0
vsetvli x28, x0, e32, m1, tu, mu
vfdiv.vf v8, v8, ft0
vsetvli x28, x0, e32, m2, tu, mu
vfdiv.vf v8, v8, ft0
vsetvli x28, x0, e32, m4, tu, mu
vfdiv.vf v8, v8, ft0
vsetvli x28, x0, e32, m8, tu, mu
vfdiv.vf v8, v8, ft0
vsetvli x28, x0, e64, m1, tu, mu
vfdiv.vf v8, v8, ft0
vsetvli x28, x0, e64, m2, tu, mu
vfdiv.vf v8, v8, ft0
vsetvli x28, x0, e64, m4, tu, mu
vfdiv.vf v8, v8, ft0
vsetvli x28, x0, e64, m8, tu, mu
vfdiv.vf v8, v8, ft0

vsetvli x28, x0, e16, mf2, tu, mu
vfmax.vv v8, v8, v8
vsetvli x28, x0, e16, mf4, tu, mu
vfmax.vv v8, v8, v8
vsetvli x28, x0, e16, m1, tu, mu
vfmax.vv v8, v8, v8
vsetvli x28, x0, e16, m2, tu, mu
vfmax.vv v8, v8, v8
vsetvli x28, x0, e16, m4, tu, mu
vfmax.vv v8, v8, v8
vsetvli x28, x0, e16, m8, tu, mu
vfmax.vv v8, v8, v8
vsetvli x28, x0, e32, mf2, tu, mu
vfmax.vv v8, v8, v8
vsetvli x28, x0, e32, m1, tu, mu
vfmax.vv v8, v8, v8
vsetvli x28, x0, e32, m2, tu, mu
vfmax.vv v8, v8, v8
vsetvli x28, x0, e32, m4, tu, mu
vfmax.vv v8, v8, v8
vsetvli x28, x0, e32, m8, tu, mu
vfmax.vv v8, v8, v8
vsetvli x28, x0, e64, m1, tu, mu
vfmax.vv v8, v8, v8
vsetvli x28, x0, e64, m2, tu, mu
vfmax.vv v8, v8, v8
vsetvli x28, x0, e64, m4, tu, mu
vfmax.vv v8, v8, v8
vsetvli x28, x0, e64, m8, tu, mu
vfmax.vv v8, v8, v8

vsetvli x28, x0, e16, mf2, tu, mu
vfmax.vf v8, v8, ft0
vsetvli x28, x0, e16, mf4, tu, mu
vfmax.vf v8, v8, ft0
vsetvli x28, x0, e16, m1, tu, mu
vfmax.vf v8, v8, ft0
vsetvli x28, x0, e16, m2, tu, mu
vfmax.vf v8, v8, ft0
vsetvli x28, x0, e16, m4, tu, mu
vfmax.vf v8, v8, ft0
vsetvli x28, x0, e16, m8, tu, mu
vfmax.vf v8, v8, ft0
vsetvli x28, x0, e32, mf2, tu, mu
vfmax.vf v8, v8, ft0
vsetvli x28, x0, e32, m1, tu, mu
vfmax.vf v8, v8, ft0
vsetvli x28, x0, e32, m2, tu, mu
vfmax.vf v8, v8, ft0
vsetvli x28, x0, e32, m4, tu, mu
vfmax.vf v8, v8, ft0
vsetvli x28, x0, e32, m8, tu, mu
vfmax.vf v8, v8, ft0
vsetvli x28, x0, e64, m1, tu, mu
vfmax.vf v8, v8, ft0
vsetvli x28, x0, e64, m2, tu, mu
vfmax.vf v8, v8, ft0
vsetvli x28, x0, e64, m4, tu, mu
vfmax.vf v8, v8, ft0
vsetvli x28, x0, e64, m8, tu, mu
vfmax.vf v8, v8, ft0

vsetvli x28, x0, e16, mf2, tu, mu
vfmin.vv v8, v8, v8
vsetvli x28, x0, e16, mf4, tu, mu
vfmin.vv v8, v8, v8
vsetvli x28, x0, e16, m1, tu, mu
vfmin.vv v8, v8, v8
vsetvli x28, x0, e16, m2, tu, mu
vfmin.vv v8, v8, v8
vsetvli x28, x0, e16, m4, tu, mu
vfmin.vv v8, v8, v8
vsetvli x28, x0, e16, m8, tu, mu
vfmin.vv v8, v8, v8
vsetvli x28, x0, e32, mf2, tu, mu
vfmin.vv v8, v8, v8
vsetvli x28, x0, e32, m1, tu, mu
vfmin.vv v8, v8, v8
vsetvli x28, x0, e32, m2, tu, mu
vfmin.vv v8, v8, v8
vsetvli x28, x0, e32, m4, tu, mu
vfmin.vv v8, v8, v8
vsetvli x28, x0, e32, m8, tu, mu
vfmin.vv v8, v8, v8
vsetvli x28, x0, e64, m1, tu, mu
vfmin.vv v8, v8, v8
vsetvli x28, x0, e64, m2, tu, mu
vfmin.vv v8, v8, v8
vsetvli x28, x0, e64, m4, tu, mu
vfmin.vv v8, v8, v8
vsetvli x28, x0, e64, m8, tu, mu
vfmin.vv v8, v8, v8

vsetvli x28, x0, e16, mf2, tu, mu
vfmin.vf v8, v8, ft0
vsetvli x28, x0, e16, mf4, tu, mu
vfmin.vf v8, v8, ft0
vsetvli x28, x0, e16, m1, tu, mu
vfmin.vf v8, v8, ft0
vsetvli x28, x0, e16, m2, tu, mu
vfmin.vf v8, v8, ft0
vsetvli x28, x0, e16, m4, tu, mu
vfmin.vf v8, v8, ft0
vsetvli x28, x0, e16, m8, tu, mu
vfmin.vf v8, v8, ft0
vsetvli x28, x0, e32, mf2, tu, mu
vfmin.vf v8, v8, ft0
vsetvli x28, x0, e32, m1, tu, mu
vfmin.vf v8, v8, ft0
vsetvli x28, x0, e32, m2, tu, mu
vfmin.vf v8, v8, ft0
vsetvli x28, x0, e32, m4, tu, mu
vfmin.vf v8, v8, ft0
vsetvli x28, x0, e32, m8, tu, mu
vfmin.vf v8, v8, ft0
vsetvli x28, x0, e64, m1, tu, mu
vfmin.vf v8, v8, ft0
vsetvli x28, x0, e64, m2, tu, mu
vfmin.vf v8, v8, ft0
vsetvli x28, x0, e64, m4, tu, mu
vfmin.vf v8, v8, ft0
vsetvli x28, x0, e64, m8, tu, mu
vfmin.vf v8, v8, ft0

vsetvli x28, x0, e16, mf2, tu, mu
vfmsac.vv v8, v8, v8
vsetvli x28, x0, e16, mf4, tu, mu
vfmsac.vv v8, v8, v8
vsetvli x28, x0, e16, m1, tu, mu
vfmsac.vv v8, v8, v8
vsetvli x28, x0, e16, m2, tu, mu
vfmsac.vv v8, v8, v8
vsetvli x28, x0, e16, m4, tu, mu
vfmsac.vv v8, v8, v8
vsetvli x28, x0, e16, m8, tu, mu
vfmsac.vv v8, v8, v8
vsetvli x28, x0, e32, mf2, tu, mu
vfmsac.vv v8, v8, v8
vsetvli x28, x0, e32, m1, tu, mu
vfmsac.vv v8, v8, v8
vsetvli x28, x0, e32, m2, tu, mu
vfmsac.vv v8, v8, v8
vsetvli x28, x0, e32, m4, tu, mu
vfmsac.vv v8, v8, v8
vsetvli x28, x0, e32, m8, tu, mu
vfmsac.vv v8, v8, v8
vsetvli x28, x0, e64, m1, tu, mu
vfmsac.vv v8, v8, v8
vsetvli x28, x0, e64, m2, tu, mu
vfmsac.vv v8, v8, v8
vsetvli x28, x0, e64, m4, tu, mu
vfmsac.vv v8, v8, v8
vsetvli x28, x0, e64, m8, tu, mu
vfmsac.vv v8, v8, v8

vsetvli x28, x0, e16, mf2, tu, mu
vfmsac.vf v8, f8, v8
vsetvli x28, x0, e16, mf4, tu, mu
vfmsac.vf v8, f8, v8
vsetvli x28, x0, e16, m1, tu, mu
vfmsac.vf v8, f8, v8
vsetvli x28, x0, e16, m2, tu, mu
vfmsac.vf v8, f8, v8
vsetvli x28, x0, e16, m4, tu, mu
vfmsac.vf v8, f8, v8
vsetvli x28, x0, e16, m8, tu, mu
vfmsac.vf v8, f8, v8
vsetvli x28, x0, e32, mf2, tu, mu
vfmsac.vf v8, f8, v8
vsetvli x28, x0, e32, m1, tu, mu
vfmsac.vf v8, f8, v8
vsetvli x28, x0, e32, m2, tu, mu
vfmsac.vf v8, f8, v8
vsetvli x28, x0, e32, m4, tu, mu
vfmsac.vf v8, f8, v8
vsetvli x28, x0, e32, m8, tu, mu
vfmsac.vf v8, f8, v8
vsetvli x28, x0, e64, m1, tu, mu
vfmsac.vf v8, f8, v8
vsetvli x28, x0, e64, m2, tu, mu
vfmsac.vf v8, f8, v8
vsetvli x28, x0, e64, m4, tu, mu
vfmsac.vf v8, f8, v8
vsetvli x28, x0, e64, m8, tu, mu
vfmsac.vf v8, f8, v8

vsetvli x28, x0, e16, mf2, tu, mu
vfmsub.vv v8, v8, v8
vsetvli x28, x0, e16, mf4, tu, mu
vfmsub.vv v8, v8, v8
vsetvli x28, x0, e16, m1, tu, mu
vfmsub.vv v8, v8, v8
vsetvli x28, x0, e16, m2, tu, mu
vfmsub.vv v8, v8, v8
vsetvli x28, x0, e16, m4, tu, mu
vfmsub.vv v8, v8, v8
vsetvli x28, x0, e16, m8, tu, mu
vfmsub.vv v8, v8, v8
vsetvli x28, x0, e32, mf2, tu, mu
vfmsub.vv v8, v8, v8
vsetvli x28, x0, e32, m1, tu, mu
vfmsub.vv v8, v8, v8
vsetvli x28, x0, e32, m2, tu, mu
vfmsub.vv v8, v8, v8
vsetvli x28, x0, e32, m4, tu, mu
vfmsub.vv v8, v8, v8
vsetvli x28, x0, e32, m8, tu, mu
vfmsub.vv v8, v8, v8
vsetvli x28, x0, e64, m1, tu, mu
vfmsub.vv v8, v8, v8
vsetvli x28, x0, e64, m2, tu, mu
vfmsub.vv v8, v8, v8
vsetvli x28, x0, e64, m4, tu, mu
vfmsub.vv v8, v8, v8
vsetvli x28, x0, e64, m8, tu, mu
vfmsub.vv v8, v8, v8

vsetvli x28, x0, e16, mf2, tu, mu
vfmsub.vf v8, f8, v8
vsetvli x28, x0, e16, mf4, tu, mu
vfmsub.vf v8, f8, v8
vsetvli x28, x0, e16, m1, tu, mu
vfmsub.vf v8, f8, v8
vsetvli x28, x0, e16, m2, tu, mu
vfmsub.vf v8, f8, v8
vsetvli x28, x0, e16, m4, tu, mu
vfmsub.vf v8, f8, v8
vsetvli x28, x0, e16, m8, tu, mu
vfmsub.vf v8, f8, v8
vsetvli x28, x0, e32, mf2, tu, mu
vfmsub.vf v8, f8, v8
vsetvli x28, x0, e32, m1, tu, mu
vfmsub.vf v8, f8, v8
vsetvli x28, x0, e32, m2, tu, mu
vfmsub.vf v8, f8, v8
vsetvli x28, x0, e32, m4, tu, mu
vfmsub.vf v8, f8, v8
vsetvli x28, x0, e32, m8, tu, mu
vfmsub.vf v8, f8, v8
vsetvli x28, x0, e64, m1, tu, mu
vfmsub.vf v8, f8, v8
vsetvli x28, x0, e64, m2, tu, mu
vfmsub.vf v8, f8, v8
vsetvli x28, x0, e64, m4, tu, mu
vfmsub.vf v8, f8, v8
vsetvli x28, x0, e64, m8, tu, mu
vfmsub.vf v8, f8, v8

vsetvli x28, x0, e16, mf2, tu, mu
vfmul.vv v8, v8, v8
vsetvli x28, x0, e16, mf4, tu, mu
vfmul.vv v8, v8, v8
vsetvli x28, x0, e16, m1, tu, mu
vfmul.vv v8, v8, v8
vsetvli x28, x0, e16, m2, tu, mu
vfmul.vv v8, v8, v8
vsetvli x28, x0, e16, m4, tu, mu
vfmul.vv v8, v8, v8
vsetvli x28, x0, e16, m8, tu, mu
vfmul.vv v8, v8, v8
vsetvli x28, x0, e32, mf2, tu, mu
vfmul.vv v8, v8, v8
vsetvli x28, x0, e32, m1, tu, mu
vfmul.vv v8, v8, v8
vsetvli x28, x0, e32, m2, tu, mu
vfmul.vv v8, v8, v8
vsetvli x28, x0, e32, m4, tu, mu
vfmul.vv v8, v8, v8
vsetvli x28, x0, e32, m8, tu, mu
vfmul.vv v8, v8, v8
vsetvli x28, x0, e64, m1, tu, mu
vfmul.vv v8, v8, v8
vsetvli x28, x0, e64, m2, tu, mu
vfmul.vv v8, v8, v8
vsetvli x28, x0, e64, m4, tu, mu
vfmul.vv v8, v8, v8
vsetvli x28, x0, e64, m8, tu, mu
vfmul.vv v8, v8, v8

vsetvli x28, x0, e16, mf2, tu, mu
vfmul.vf v8, v8, ft0
vsetvli x28, x0, e16, mf4, tu, mu
vfmul.vf v8, v8, ft0
vsetvli x28, x0, e16, m1, tu, mu
vfmul.vf v8, v8, ft0
vsetvli x28, x0, e16, m2, tu, mu
vfmul.vf v8, v8, ft0
vsetvli x28, x0, e16, m4, tu, mu
vfmul.vf v8, v8, ft0
vsetvli x28, x0, e16, m8, tu, mu
vfmul.vf v8, v8, ft0
vsetvli x28, x0, e32, mf2, tu, mu
vfmul.vf v8, v8, ft0
vsetvli x28, x0, e32, m1, tu, mu
vfmul.vf v8, v8, ft0
vsetvli x28, x0, e32, m2, tu, mu
vfmul.vf v8, v8, ft0
vsetvli x28, x0, e32, m4, tu, mu
vfmul.vf v8, v8, ft0
vsetvli x28, x0, e32, m8, tu, mu
vfmul.vf v8, v8, ft0
vsetvli x28, x0, e64, m1, tu, mu
vfmul.vf v8, v8, ft0
vsetvli x28, x0, e64, m2, tu, mu
vfmul.vf v8, v8, ft0
vsetvli x28, x0, e64, m4, tu, mu
vfmul.vf v8, v8, ft0
vsetvli x28, x0, e64, m8, tu, mu
vfmul.vf v8, v8, ft0

vsetvli x28, x0, e16, mf2, tu, mu
vfmacc.vf v8, f8, v8
vsetvli x28, x0, e16, mf4, tu, mu
vfmacc.vf v8, f8, v8
vsetvli x28, x0, e16, m1, tu, mu
vfmacc.vf v8, f8, v8
vsetvli x28, x0, e16, m2, tu, mu
vfmacc.vf v8, f8, v8
vsetvli x28, x0, e16, m4, tu, mu
vfmacc.vf v8, f8, v8
vsetvli x28, x0, e16, m8, tu, mu
vfmacc.vf v8, f8, v8
vsetvli x28, x0, e32, mf2, tu, mu
vfmacc.vf v8, f8, v8
vsetvli x28, x0, e32, m1, tu, mu
vfmacc.vf v8, f8, v8
vsetvli x28, x0, e32, m2, tu, mu
vfmacc.vf v8, f8, v8
vsetvli x28, x0, e32, m4, tu, mu
vfmacc.vf v8, f8, v8
vsetvli x28, x0, e32, m8, tu, mu
vfmacc.vf v8, f8, v8
vsetvli x28, x0, e64, m1, tu, mu
vfmacc.vf v8, f8, v8
vsetvli x28, x0, e64, m2, tu, mu
vfmacc.vf v8, f8, v8
vsetvli x28, x0, e64, m4, tu, mu
vfmacc.vf v8, f8, v8
vsetvli x28, x0, e64, m8, tu, mu
vfmacc.vf v8, f8, v8

vsetvli x28, x0, e16, mf2, tu, mu
vfmacc.vv v8, v8, v8
vsetvli x28, x0, e16, mf4, tu, mu
vfmacc.vv v8, v8, v8
vsetvli x28, x0, e16, m1, tu, mu
vfmacc.vv v8, v8, v8
vsetvli x28, x0, e16, m2, tu, mu
vfmacc.vv v8, v8, v8
vsetvli x28, x0, e16, m4, tu, mu
vfmacc.vv v8, v8, v8
vsetvli x28, x0, e16, m8, tu, mu
vfmacc.vv v8, v8, v8
vsetvli x28, x0, e32, mf2, tu, mu
vfmacc.vv v8, v8, v8
vsetvli x28, x0, e32, m1, tu, mu
vfmacc.vv v8, v8, v8
vsetvli x28, x0, e32, m2, tu, mu
vfmacc.vv v8, v8, v8
vsetvli x28, x0, e32, m4, tu, mu
vfmacc.vv v8, v8, v8
vsetvli x28, x0, e32, m8, tu, mu
vfmacc.vv v8, v8, v8
vsetvli x28, x0, e64, m1, tu, mu
vfmacc.vv v8, v8, v8
vsetvli x28, x0, e64, m2, tu, mu
vfmacc.vv v8, v8, v8
vsetvli x28, x0, e64, m4, tu, mu
vfmacc.vv v8, v8, v8
vsetvli x28, x0, e64, m8, tu, mu
vfmacc.vv v8, v8, v8

vsetvli x28, x0, e16, mf2, tu, mu
vfmadd.vf v8, f8, v8
vsetvli x28, x0, e16, mf4, tu, mu
vfmadd.vf v8, f8, v8
vsetvli x28, x0, e16, m1, tu, mu
vfmadd.vf v8, f8, v8
vsetvli x28, x0, e16, m2, tu, mu
vfmadd.vf v8, f8, v8
vsetvli x28, x0, e16, m4, tu, mu
vfmadd.vf v8, f8, v8
vsetvli x28, x0, e16, m8, tu, mu
vfmadd.vf v8, f8, v8
vsetvli x28, x0, e32, mf2, tu, mu
vfmadd.vf v8, f8, v8
vsetvli x28, x0, e32, m1, tu, mu
vfmadd.vf v8, f8, v8
vsetvli x28, x0, e32, m2, tu, mu
vfmadd.vf v8, f8, v8
vsetvli x28, x0, e32, m4, tu, mu
vfmadd.vf v8, f8, v8
vsetvli x28, x0, e32, m8, tu, mu
vfmadd.vf v8, f8, v8
vsetvli x28, x0, e64, m1, tu, mu
vfmadd.vf v8, f8, v8
vsetvli x28, x0, e64, m2, tu, mu
vfmadd.vf v8, f8, v8
vsetvli x28, x0, e64, m4, tu, mu
vfmadd.vf v8, f8, v8
vsetvli x28, x0, e64, m8, tu, mu
vfmadd.vf v8, f8, v8

vsetvli x28, x0, e16, mf2, tu, mu
vfmadd.vv v8, v8, v8
vsetvli x28, x0, e16, mf4, tu, mu
vfmadd.vv v8, v8, v8
vsetvli x28, x0, e16, m1, tu, mu
vfmadd.vv v8, v8, v8
vsetvli x28, x0, e16, m2, tu, mu
vfmadd.vv v8, v8, v8
vsetvli x28, x0, e16, m4, tu, mu
vfmadd.vv v8, v8, v8
vsetvli x28, x0, e16, m8, tu, mu
vfmadd.vv v8, v8, v8
vsetvli x28, x0, e32, mf2, tu, mu
vfmadd.vv v8, v8, v8
vsetvli x28, x0, e32, m1, tu, mu
vfmadd.vv v8, v8, v8
vsetvli x28, x0, e32, m2, tu, mu
vfmadd.vv v8, v8, v8
vsetvli x28, x0, e32, m4, tu, mu
vfmadd.vv v8, v8, v8
vsetvli x28, x0, e32, m8, tu, mu
vfmadd.vv v8, v8, v8
vsetvli x28, x0, e64, m1, tu, mu
vfmadd.vv v8, v8, v8
vsetvli x28, x0, e64, m2, tu, mu
vfmadd.vv v8, v8, v8
vsetvli x28, x0, e64, m4, tu, mu
vfmadd.vv v8, v8, v8
vsetvli x28, x0, e64, m8, tu, mu
vfmadd.vv v8, v8, v8

vsetvli x28, x0, e16, mf2, tu, mu
vfmv.f.s f8, v8
vsetvli x28, x0, e16, mf4, tu, mu
vfmv.f.s f8, v8
vsetvli x28, x0, e16, m1, tu, mu
vfmv.f.s f8, v8
vsetvli x28, x0, e16, m2, tu, mu
vfmv.f.s f8, v8
vsetvli x28, x0, e16, m4, tu, mu
vfmv.f.s f8, v8
vsetvli x28, x0, e16, m8, tu, mu
vfmv.f.s f8, v8
vsetvli x28, x0, e32, mf2, tu, mu
vfmv.f.s f8, v8
vsetvli x28, x0, e32, m1, tu, mu
vfmv.f.s f8, v8
vsetvli x28, x0, e32, m2, tu, mu
vfmv.f.s f8, v8
vsetvli x28, x0, e32, m4, tu, mu
vfmv.f.s f8, v8
vsetvli x28, x0, e32, m8, tu, mu
vfmv.f.s f8, v8
vsetvli x28, x0, e64, m1, tu, mu
vfmv.f.s f8, v8
vsetvli x28, x0, e64, m2, tu, mu
vfmv.f.s f8, v8
vsetvli x28, x0, e64, m4, tu, mu
vfmv.f.s f8, v8
vsetvli x28, x0, e64, m8, tu, mu
vfmv.f.s f8, v8

vsetvli x28, x0, e16, mf2, tu, mu
vfmv.s.f v8, f8
vsetvli x28, x0, e16, mf4, tu, mu
vfmv.s.f v8, f8
vsetvli x28, x0, e16, m1, tu, mu
vfmv.s.f v8, f8
vsetvli x28, x0, e16, m2, tu, mu
vfmv.s.f v8, f8
vsetvli x28, x0, e16, m4, tu, mu
vfmv.s.f v8, f8
vsetvli x28, x0, e16, m8, tu, mu
vfmv.s.f v8, f8
vsetvli x28, x0, e32, mf2, tu, mu
vfmv.s.f v8, f8
vsetvli x28, x0, e32, m1, tu, mu
vfmv.s.f v8, f8
vsetvli x28, x0, e32, m2, tu, mu
vfmv.s.f v8, f8
vsetvli x28, x0, e32, m4, tu, mu
vfmv.s.f v8, f8
vsetvli x28, x0, e32, m8, tu, mu
vfmv.s.f v8, f8
vsetvli x28, x0, e64, m1, tu, mu
vfmv.s.f v8, f8
vsetvli x28, x0, e64, m2, tu, mu
vfmv.s.f v8, f8
vsetvli x28, x0, e64, m4, tu, mu
vfmv.s.f v8, f8
vsetvli x28, x0, e64, m8, tu, mu
vfmv.s.f v8, f8

vsetvli x28, x0, e16, mf2, tu, mu
vfmv.v.f v8, f8
vsetvli x28, x0, e16, mf4, tu, mu
vfmv.v.f v8, f8
vsetvli x28, x0, e16, m1, tu, mu
vfmv.v.f v8, f8
vsetvli x28, x0, e16, m2, tu, mu
vfmv.v.f v8, f8
vsetvli x28, x0, e16, m4, tu, mu
vfmv.v.f v8, f8
vsetvli x28, x0, e16, m8, tu, mu
vfmv.v.f v8, f8
vsetvli x28, x0, e32, mf2, tu, mu
vfmv.v.f v8, f8
vsetvli x28, x0, e32, m1, tu, mu
vfmv.v.f v8, f8
vsetvli x28, x0, e32, m2, tu, mu
vfmv.v.f v8, f8
vsetvli x28, x0, e32, m4, tu, mu
vfmv.v.f v8, f8
vsetvli x28, x0, e32, m8, tu, mu
vfmv.v.f v8, f8
vsetvli x28, x0, e64, m1, tu, mu
vfmv.v.f v8, f8
vsetvli x28, x0, e64, m2, tu, mu
vfmv.v.f v8, f8
vsetvli x28, x0, e64, m4, tu, mu
vfmv.v.f v8, f8
vsetvli x28, x0, e64, m8, tu, mu
vfmv.v.f v8, f8

vsetvli x28, x0, e16, mf2, tu, mu
vfnmacc.vf v8, f8, v8
vsetvli x28, x0, e16, mf4, tu, mu
vfnmacc.vf v8, f8, v8
vsetvli x28, x0, e16, m1, tu, mu
vfnmacc.vf v8, f8, v8
vsetvli x28, x0, e16, m2, tu, mu
vfnmacc.vf v8, f8, v8
vsetvli x28, x0, e16, m4, tu, mu
vfnmacc.vf v8, f8, v8
vsetvli x28, x0, e16, m8, tu, mu
vfnmacc.vf v8, f8, v8
vsetvli x28, x0, e32, mf2, tu, mu
vfnmacc.vf v8, f8, v8
vsetvli x28, x0, e32, m1, tu, mu
vfnmacc.vf v8, f8, v8
vsetvli x28, x0, e32, m2, tu, mu
vfnmacc.vf v8, f8, v8
vsetvli x28, x0, e32, m4, tu, mu
vfnmacc.vf v8, f8, v8
vsetvli x28, x0, e32, m8, tu, mu
vfnmacc.vf v8, f8, v8
vsetvli x28, x0, e64, m1, tu, mu
vfnmacc.vf v8, f8, v8
vsetvli x28, x0, e64, m2, tu, mu
vfnmacc.vf v8, f8, v8
vsetvli x28, x0, e64, m4, tu, mu
vfnmacc.vf v8, f8, v8
vsetvli x28, x0, e64, m8, tu, mu
vfnmacc.vf v8, f8, v8

vsetvli x28, x0, e16, mf2, tu, mu
vfnmacc.vv v8, v8, v8
vsetvli x28, x0, e16, mf4, tu, mu
vfnmacc.vv v8, v8, v8
vsetvli x28, x0, e16, m1, tu, mu
vfnmacc.vv v8, v8, v8
vsetvli x28, x0, e16, m2, tu, mu
vfnmacc.vv v8, v8, v8
vsetvli x28, x0, e16, m4, tu, mu
vfnmacc.vv v8, v8, v8
vsetvli x28, x0, e16, m8, tu, mu
vfnmacc.vv v8, v8, v8
vsetvli x28, x0, e32, mf2, tu, mu
vfnmacc.vv v8, v8, v8
vsetvli x28, x0, e32, m1, tu, mu
vfnmacc.vv v8, v8, v8
vsetvli x28, x0, e32, m2, tu, mu
vfnmacc.vv v8, v8, v8
vsetvli x28, x0, e32, m4, tu, mu
vfnmacc.vv v8, v8, v8
vsetvli x28, x0, e32, m8, tu, mu
vfnmacc.vv v8, v8, v8
vsetvli x28, x0, e64, m1, tu, mu
vfnmacc.vv v8, v8, v8
vsetvli x28, x0, e64, m2, tu, mu
vfnmacc.vv v8, v8, v8
vsetvli x28, x0, e64, m4, tu, mu
vfnmacc.vv v8, v8, v8
vsetvli x28, x0, e64, m8, tu, mu
vfnmacc.vv v8, v8, v8

vsetvli x28, x0, e16, mf2, tu, mu
vfnmadd.vf v8, f8, v8
vsetvli x28, x0, e16, mf4, tu, mu
vfnmadd.vf v8, f8, v8
vsetvli x28, x0, e16, m1, tu, mu
vfnmadd.vf v8, f8, v8
vsetvli x28, x0, e16, m2, tu, mu
vfnmadd.vf v8, f8, v8
vsetvli x28, x0, e16, m4, tu, mu
vfnmadd.vf v8, f8, v8
vsetvli x28, x0, e16, m8, tu, mu
vfnmadd.vf v8, f8, v8
vsetvli x28, x0, e32, mf2, tu, mu
vfnmadd.vf v8, f8, v8
vsetvli x28, x0, e32, m1, tu, mu
vfnmadd.vf v8, f8, v8
vsetvli x28, x0, e32, m2, tu, mu
vfnmadd.vf v8, f8, v8
vsetvli x28, x0, e32, m4, tu, mu
vfnmadd.vf v8, f8, v8
vsetvli x28, x0, e32, m8, tu, mu
vfnmadd.vf v8, f8, v8
vsetvli x28, x0, e64, m1, tu, mu
vfnmadd.vf v8, f8, v8
vsetvli x28, x0, e64, m2, tu, mu
vfnmadd.vf v8, f8, v8
vsetvli x28, x0, e64, m4, tu, mu
vfnmadd.vf v8, f8, v8
vsetvli x28, x0, e64, m8, tu, mu
vfnmadd.vf v8, f8, v8

vsetvli x28, x0, e16, mf2, tu, mu
vfnmadd.vv v8, v8, v8
vsetvli x28, x0, e16, mf4, tu, mu
vfnmadd.vv v8, v8, v8
vsetvli x28, x0, e16, m1, tu, mu
vfnmadd.vv v8, v8, v8
vsetvli x28, x0, e16, m2, tu, mu
vfnmadd.vv v8, v8, v8
vsetvli x28, x0, e16, m4, tu, mu
vfnmadd.vv v8, v8, v8
vsetvli x28, x0, e16, m8, tu, mu
vfnmadd.vv v8, v8, v8
vsetvli x28, x0, e32, mf2, tu, mu
vfnmadd.vv v8, v8, v8
vsetvli x28, x0, e32, m1, tu, mu
vfnmadd.vv v8, v8, v8
vsetvli x28, x0, e32, m2, tu, mu
vfnmadd.vv v8, v8, v8
vsetvli x28, x0, e32, m4, tu, mu
vfnmadd.vv v8, v8, v8
vsetvli x28, x0, e32, m8, tu, mu
vfnmadd.vv v8, v8, v8
vsetvli x28, x0, e64, m1, tu, mu
vfnmadd.vv v8, v8, v8
vsetvli x28, x0, e64, m2, tu, mu
vfnmadd.vv v8, v8, v8
vsetvli x28, x0, e64, m4, tu, mu
vfnmadd.vv v8, v8, v8
vsetvli x28, x0, e64, m8, tu, mu
vfnmadd.vv v8, v8, v8

vsetvli x28, x0, e16, mf2, tu, mu
vfnmsac.vf v8, f8, v8
vsetvli x28, x0, e16, mf4, tu, mu
vfnmsac.vf v8, f8, v8
vsetvli x28, x0, e16, m1, tu, mu
vfnmsac.vf v8, f8, v8
vsetvli x28, x0, e16, m2, tu, mu
vfnmsac.vf v8, f8, v8
vsetvli x28, x0, e16, m4, tu, mu
vfnmsac.vf v8, f8, v8
vsetvli x28, x0, e16, m8, tu, mu
vfnmsac.vf v8, f8, v8
vsetvli x28, x0, e32, mf2, tu, mu
vfnmsac.vf v8, f8, v8
vsetvli x28, x0, e32, m1, tu, mu
vfnmsac.vf v8, f8, v8
vsetvli x28, x0, e32, m2, tu, mu
vfnmsac.vf v8, f8, v8
vsetvli x28, x0, e32, m4, tu, mu
vfnmsac.vf v8, f8, v8
vsetvli x28, x0, e32, m8, tu, mu
vfnmsac.vf v8, f8, v8
vsetvli x28, x0, e64, m1, tu, mu
vfnmsac.vf v8, f8, v8
vsetvli x28, x0, e64, m2, tu, mu
vfnmsac.vf v8, f8, v8
vsetvli x28, x0, e64, m4, tu, mu
vfnmsac.vf v8, f8, v8
vsetvli x28, x0, e64, m8, tu, mu
vfnmsac.vf v8, f8, v8

vsetvli x28, x0, e16, mf2, tu, mu
vfnmsac.vv v8, v8, v8
vsetvli x28, x0, e16, mf4, tu, mu
vfnmsac.vv v8, v8, v8
vsetvli x28, x0, e16, m1, tu, mu
vfnmsac.vv v8, v8, v8
vsetvli x28, x0, e16, m2, tu, mu
vfnmsac.vv v8, v8, v8
vsetvli x28, x0, e16, m4, tu, mu
vfnmsac.vv v8, v8, v8
vsetvli x28, x0, e16, m8, tu, mu
vfnmsac.vv v8, v8, v8
vsetvli x28, x0, e32, mf2, tu, mu
vfnmsac.vv v8, v8, v8
vsetvli x28, x0, e32, m1, tu, mu
vfnmsac.vv v8, v8, v8
vsetvli x28, x0, e32, m2, tu, mu
vfnmsac.vv v8, v8, v8
vsetvli x28, x0, e32, m4, tu, mu
vfnmsac.vv v8, v8, v8
vsetvli x28, x0, e32, m8, tu, mu
vfnmsac.vv v8, v8, v8
vsetvli x28, x0, e64, m1, tu, mu
vfnmsac.vv v8, v8, v8
vsetvli x28, x0, e64, m2, tu, mu
vfnmsac.vv v8, v8, v8
vsetvli x28, x0, e64, m4, tu, mu
vfnmsac.vv v8, v8, v8
vsetvli x28, x0, e64, m8, tu, mu
vfnmsac.vv v8, v8, v8

vsetvli x28, x0, e16, mf2, tu, mu
vfnmsub.vf v8, f8, v8
vsetvli x28, x0, e16, mf4, tu, mu
vfnmsub.vf v8, f8, v8
vsetvli x28, x0, e16, m1, tu, mu
vfnmsub.vf v8, f8, v8
vsetvli x28, x0, e16, m2, tu, mu
vfnmsub.vf v8, f8, v8
vsetvli x28, x0, e16, m4, tu, mu
vfnmsub.vf v8, f8, v8
vsetvli x28, x0, e16, m8, tu, mu
vfnmsub.vf v8, f8, v8
vsetvli x28, x0, e32, mf2, tu, mu
vfnmsub.vf v8, f8, v8
vsetvli x28, x0, e32, m1, tu, mu
vfnmsub.vf v8, f8, v8
vsetvli x28, x0, e32, m2, tu, mu
vfnmsub.vf v8, f8, v8
vsetvli x28, x0, e32, m4, tu, mu
vfnmsub.vf v8, f8, v8
vsetvli x28, x0, e32, m8, tu, mu
vfnmsub.vf v8, f8, v8
vsetvli x28, x0, e64, m1, tu, mu
vfnmsub.vf v8, f8, v8
vsetvli x28, x0, e64, m2, tu, mu
vfnmsub.vf v8, f8, v8
vsetvli x28, x0, e64, m4, tu, mu
vfnmsub.vf v8, f8, v8
vsetvli x28, x0, e64, m8, tu, mu
vfnmsub.vf v8, f8, v8

vsetvli x28, x0, e16, mf2, tu, mu
vfnmsub.vv v8, v8, v8
vsetvli x28, x0, e16, mf4, tu, mu
vfnmsub.vv v8, v8, v8
vsetvli x28, x0, e16, m1, tu, mu
vfnmsub.vv v8, v8, v8
vsetvli x28, x0, e16, m2, tu, mu
vfnmsub.vv v8, v8, v8
vsetvli x28, x0, e16, m4, tu, mu
vfnmsub.vv v8, v8, v8
vsetvli x28, x0, e16, m8, tu, mu
vfnmsub.vv v8, v8, v8
vsetvli x28, x0, e32, mf2, tu, mu
vfnmsub.vv v8, v8, v8
vsetvli x28, x0, e32, m1, tu, mu
vfnmsub.vv v8, v8, v8
vsetvli x28, x0, e32, m2, tu, mu
vfnmsub.vv v8, v8, v8
vsetvli x28, x0, e32, m4, tu, mu
vfnmsub.vv v8, v8, v8
vsetvli x28, x0, e32, m8, tu, mu
vfnmsub.vv v8, v8, v8
vsetvli x28, x0, e64, m1, tu, mu
vfnmsub.vv v8, v8, v8
vsetvli x28, x0, e64, m2, tu, mu
vfnmsub.vv v8, v8, v8
vsetvli x28, x0, e64, m4, tu, mu
vfnmsub.vv v8, v8, v8
vsetvli x28, x0, e64, m8, tu, mu
vfnmsub.vv v8, v8, v8

vsetvli x28, x0, e16, mf2, tu, mu
vfrdiv.vf v8, v8, ft0
vsetvli x28, x0, e16, mf4, tu, mu
vfrdiv.vf v8, v8, ft0
vsetvli x28, x0, e16, m1, tu, mu
vfrdiv.vf v8, v8, ft0
vsetvli x28, x0, e16, m2, tu, mu
vfrdiv.vf v8, v8, ft0
vsetvli x28, x0, e16, m4, tu, mu
vfrdiv.vf v8, v8, ft0
vsetvli x28, x0, e16, m8, tu, mu
vfrdiv.vf v8, v8, ft0
vsetvli x28, x0, e32, mf2, tu, mu
vfrdiv.vf v8, v8, ft0
vsetvli x28, x0, e32, m1, tu, mu
vfrdiv.vf v8, v8, ft0
vsetvli x28, x0, e32, m2, tu, mu
vfrdiv.vf v8, v8, ft0
vsetvli x28, x0, e32, m4, tu, mu
vfrdiv.vf v8, v8, ft0
vsetvli x28, x0, e32, m8, tu, mu
vfrdiv.vf v8, v8, ft0
vsetvli x28, x0, e64, m1, tu, mu
vfrdiv.vf v8, v8, ft0
vsetvli x28, x0, e64, m2, tu, mu
vfrdiv.vf v8, v8, ft0
vsetvli x28, x0, e64, m4, tu, mu
vfrdiv.vf v8, v8, ft0
vsetvli x28, x0, e64, m8, tu, mu
vfrdiv.vf v8, v8, ft0

vsetvli x28, x0, e16, mf2, tu, mu
vfrec7.v v8, v8
vsetvli x28, x0, e16, mf4, tu, mu
vfrec7.v v8, v8
vsetvli x28, x0, e16, m1, tu, mu
vfrec7.v v8, v8
vsetvli x28, x0, e16, m2, tu, mu
vfrec7.v v8, v8
vsetvli x28, x0, e16, m4, tu, mu
vfrec7.v v8, v8
vsetvli x28, x0, e16, m8, tu, mu
vfrec7.v v8, v8
vsetvli x28, x0, e32, mf2, tu, mu
vfrec7.v v8, v8
vsetvli x28, x0, e32, m1, tu, mu
vfrec7.v v8, v8
vsetvli x28, x0, e32, m2, tu, mu
vfrec7.v v8, v8
vsetvli x28, x0, e32, m4, tu, mu
vfrec7.v v8, v8
vsetvli x28, x0, e32, m8, tu, mu
vfrec7.v v8, v8
vsetvli x28, x0, e64, m1, tu, mu
vfrec7.v v8, v8
vsetvli x28, x0, e64, m2, tu, mu
vfrec7.v v8, v8
vsetvli x28, x0, e64, m4, tu, mu
vfrec7.v v8, v8
vsetvli x28, x0, e64, m8, tu, mu
vfrec7.v v8, v8

vsetvli x28, x0, e16, mf2, tu, mu
vfrsqrt7.v v8, v8
vsetvli x28, x0, e16, mf4, tu, mu
vfrsqrt7.v v8, v8
vsetvli x28, x0, e16, m1, tu, mu
vfrsqrt7.v v8, v8
vsetvli x28, x0, e16, m2, tu, mu
vfrsqrt7.v v8, v8
vsetvli x28, x0, e16, m4, tu, mu
vfrsqrt7.v v8, v8
vsetvli x28, x0, e16, m8, tu, mu
vfrsqrt7.v v8, v8
vsetvli x28, x0, e32, mf2, tu, mu
vfrsqrt7.v v8, v8
vsetvli x28, x0, e32, m1, tu, mu
vfrsqrt7.v v8, v8
vsetvli x28, x0, e32, m2, tu, mu
vfrsqrt7.v v8, v8
vsetvli x28, x0, e32, m4, tu, mu
vfrsqrt7.v v8, v8
vsetvli x28, x0, e32, m8, tu, mu
vfrsqrt7.v v8, v8
vsetvli x28, x0, e64, m1, tu, mu
vfrsqrt7.v v8, v8
vsetvli x28, x0, e64, m2, tu, mu
vfrsqrt7.v v8, v8
vsetvli x28, x0, e64, m4, tu, mu
vfrsqrt7.v v8, v8
vsetvli x28, x0, e64, m8, tu, mu
vfrsqrt7.v v8, v8

vsetvli x28, x0, e16, mf2, tu, mu
vfrsub.vf v8, v8, ft0
vsetvli x28, x0, e16, mf4, tu, mu
vfrsub.vf v8, v8, ft0
vsetvli x28, x0, e16, m1, tu, mu
vfrsub.vf v8, v8, ft0
vsetvli x28, x0, e16, m2, tu, mu
vfrsub.vf v8, v8, ft0
vsetvli x28, x0, e16, m4, tu, mu
vfrsub.vf v8, v8, ft0
vsetvli x28, x0, e16, m8, tu, mu
vfrsub.vf v8, v8, ft0
vsetvli x28, x0, e32, mf2, tu, mu
vfrsub.vf v8, v8, ft0
vsetvli x28, x0, e32, m1, tu, mu
vfrsub.vf v8, v8, ft0
vsetvli x28, x0, e32, m2, tu, mu
vfrsub.vf v8, v8, ft0
vsetvli x28, x0, e32, m4, tu, mu
vfrsub.vf v8, v8, ft0
vsetvli x28, x0, e32, m8, tu, mu
vfrsub.vf v8, v8, ft0
vsetvli x28, x0, e64, m1, tu, mu
vfrsub.vf v8, v8, ft0
vsetvli x28, x0, e64, m2, tu, mu
vfrsub.vf v8, v8, ft0
vsetvli x28, x0, e64, m4, tu, mu
vfrsub.vf v8, v8, ft0
vsetvli x28, x0, e64, m8, tu, mu
vfrsub.vf v8, v8, ft0

vsetvli x28, x0, e16, mf2, tu, mu
vfsqrt.v v8, v8
vsetvli x28, x0, e16, mf4, tu, mu
vfsqrt.v v8, v8
vsetvli x28, x0, e16, m1, tu, mu
vfsqrt.v v8, v8
vsetvli x28, x0, e16, m2, tu, mu
vfsqrt.v v8, v8
vsetvli x28, x0, e16, m4, tu, mu
vfsqrt.v v8, v8
vsetvli x28, x0, e16, m8, tu, mu
vfsqrt.v v8, v8
vsetvli x28, x0, e32, mf2, tu, mu
vfsqrt.v v8, v8
vsetvli x28, x0, e32, m1, tu, mu
vfsqrt.v v8, v8
vsetvli x28, x0, e32, m2, tu, mu
vfsqrt.v v8, v8
vsetvli x28, x0, e32, m4, tu, mu
vfsqrt.v v8, v8
vsetvli x28, x0, e32, m8, tu, mu
vfsqrt.v v8, v8
vsetvli x28, x0, e64, m1, tu, mu
vfsqrt.v v8, v8
vsetvli x28, x0, e64, m2, tu, mu
vfsqrt.v v8, v8
vsetvli x28, x0, e64, m4, tu, mu
vfsqrt.v v8, v8
vsetvli x28, x0, e64, m8, tu, mu
vfsqrt.v v8, v8

vsetvli x28, x0, e16, mf2, tu, mu
vfsgnjn.vf v8, v8, ft0
vsetvli x28, x0, e16, mf4, tu, mu
vfsgnjn.vf v8, v8, ft0
vsetvli x28, x0, e16, m1, tu, mu
vfsgnjn.vf v8, v8, ft0
vsetvli x28, x0, e16, m2, tu, mu
vfsgnjn.vf v8, v8, ft0
vsetvli x28, x0, e16, m4, tu, mu
vfsgnjn.vf v8, v8, ft0
vsetvli x28, x0, e16, m8, tu, mu
vfsgnjn.vf v8, v8, ft0
vsetvli x28, x0, e32, mf2, tu, mu
vfsgnjn.vf v8, v8, ft0
vsetvli x28, x0, e32, m1, tu, mu
vfsgnjn.vf v8, v8, ft0
vsetvli x28, x0, e32, m2, tu, mu
vfsgnjn.vf v8, v8, ft0
vsetvli x28, x0, e32, m4, tu, mu
vfsgnjn.vf v8, v8, ft0
vsetvli x28, x0, e32, m8, tu, mu
vfsgnjn.vf v8, v8, ft0
vsetvli x28, x0, e64, m1, tu, mu
vfsgnjn.vf v8, v8, ft0
vsetvli x28, x0, e64, m2, tu, mu
vfsgnjn.vf v8, v8, ft0
vsetvli x28, x0, e64, m4, tu, mu
vfsgnjn.vf v8, v8, ft0
vsetvli x28, x0, e64, m8, tu, mu
vfsgnjn.vf v8, v8, ft0

vsetvli x28, x0, e16, mf2, tu, mu
vfsgnjn.vv v8, v8, v8
vsetvli x28, x0, e16, mf4, tu, mu
vfsgnjn.vv v8, v8, v8
vsetvli x28, x0, e16, m1, tu, mu
vfsgnjn.vv v8, v8, v8
vsetvli x28, x0, e16, m2, tu, mu
vfsgnjn.vv v8, v8, v8
vsetvli x28, x0, e16, m4, tu, mu
vfsgnjn.vv v8, v8, v8
vsetvli x28, x0, e16, m8, tu, mu
vfsgnjn.vv v8, v8, v8
vsetvli x28, x0, e32, mf2, tu, mu
vfsgnjn.vv v8, v8, v8
vsetvli x28, x0, e32, m1, tu, mu
vfsgnjn.vv v8, v8, v8
vsetvli x28, x0, e32, m2, tu, mu
vfsgnjn.vv v8, v8, v8
vsetvli x28, x0, e32, m4, tu, mu
vfsgnjn.vv v8, v8, v8
vsetvli x28, x0, e32, m8, tu, mu
vfsgnjn.vv v8, v8, v8
vsetvli x28, x0, e64, m1, tu, mu
vfsgnjn.vv v8, v8, v8
vsetvli x28, x0, e64, m2, tu, mu
vfsgnjn.vv v8, v8, v8
vsetvli x28, x0, e64, m4, tu, mu
vfsgnjn.vv v8, v8, v8
vsetvli x28, x0, e64, m8, tu, mu
vfsgnjn.vv v8, v8, v8

vsetvli x28, x0, e16, mf2, tu, mu
vfsgnj.vf v8, v8, ft0
vsetvli x28, x0, e16, mf4, tu, mu
vfsgnj.vf v8, v8, ft0
vsetvli x28, x0, e16, m1, tu, mu
vfsgnj.vf v8, v8, ft0
vsetvli x28, x0, e16, m2, tu, mu
vfsgnj.vf v8, v8, ft0
vsetvli x28, x0, e16, m4, tu, mu
vfsgnj.vf v8, v8, ft0
vsetvli x28, x0, e16, m8, tu, mu
vfsgnj.vf v8, v8, ft0
vsetvli x28, x0, e32, mf2, tu, mu
vfsgnj.vf v8, v8, ft0
vsetvli x28, x0, e32, m1, tu, mu
vfsgnj.vf v8, v8, ft0
vsetvli x28, x0, e32, m2, tu, mu
vfsgnj.vf v8, v8, ft0
vsetvli x28, x0, e32, m4, tu, mu
vfsgnj.vf v8, v8, ft0
vsetvli x28, x0, e32, m8, tu, mu
vfsgnj.vf v8, v8, ft0
vsetvli x28, x0, e64, m1, tu, mu
vfsgnj.vf v8, v8, ft0
vsetvli x28, x0, e64, m2, tu, mu
vfsgnj.vf v8, v8, ft0
vsetvli x28, x0, e64, m4, tu, mu
vfsgnj.vf v8, v8, ft0
vsetvli x28, x0, e64, m8, tu, mu
vfsgnj.vf v8, v8, ft0

vsetvli x28, x0, e16, mf2, tu, mu
vfsgnj.vv v8, v8, v8
vsetvli x28, x0, e16, mf4, tu, mu
vfsgnj.vv v8, v8, v8
vsetvli x28, x0, e16, m1, tu, mu
vfsgnj.vv v8, v8, v8
vsetvli x28, x0, e16, m2, tu, mu
vfsgnj.vv v8, v8, v8
vsetvli x28, x0, e16, m4, tu, mu
vfsgnj.vv v8, v8, v8
vsetvli x28, x0, e16, m8, tu, mu
vfsgnj.vv v8, v8, v8
vsetvli x28, x0, e32, mf2, tu, mu
vfsgnj.vv v8, v8, v8
vsetvli x28, x0, e32, m1, tu, mu
vfsgnj.vv v8, v8, v8
vsetvli x28, x0, e32, m2, tu, mu
vfsgnj.vv v8, v8, v8
vsetvli x28, x0, e32, m4, tu, mu
vfsgnj.vv v8, v8, v8
vsetvli x28, x0, e32, m8, tu, mu
vfsgnj.vv v8, v8, v8
vsetvli x28, x0, e64, m1, tu, mu
vfsgnj.vv v8, v8, v8
vsetvli x28, x0, e64, m2, tu, mu
vfsgnj.vv v8, v8, v8
vsetvli x28, x0, e64, m4, tu, mu
vfsgnj.vv v8, v8, v8
vsetvli x28, x0, e64, m8, tu, mu
vfsgnj.vv v8, v8, v8

vsetvli x28, x0, e16, mf2, tu, mu
vfsgnjx.vf v8, v8, ft0
vsetvli x28, x0, e16, mf4, tu, mu
vfsgnjx.vf v8, v8, ft0
vsetvli x28, x0, e16, m1, tu, mu
vfsgnjx.vf v8, v8, ft0
vsetvli x28, x0, e16, m2, tu, mu
vfsgnjx.vf v8, v8, ft0
vsetvli x28, x0, e16, m4, tu, mu
vfsgnjx.vf v8, v8, ft0
vsetvli x28, x0, e16, m8, tu, mu
vfsgnjx.vf v8, v8, ft0
vsetvli x28, x0, e32, mf2, tu, mu
vfsgnjx.vf v8, v8, ft0
vsetvli x28, x0, e32, m1, tu, mu
vfsgnjx.vf v8, v8, ft0
vsetvli x28, x0, e32, m2, tu, mu
vfsgnjx.vf v8, v8, ft0
vsetvli x28, x0, e32, m4, tu, mu
vfsgnjx.vf v8, v8, ft0
vsetvli x28, x0, e32, m8, tu, mu
vfsgnjx.vf v8, v8, ft0
vsetvli x28, x0, e64, m1, tu, mu
vfsgnjx.vf v8, v8, ft0
vsetvli x28, x0, e64, m2, tu, mu
vfsgnjx.vf v8, v8, ft0
vsetvli x28, x0, e64, m4, tu, mu
vfsgnjx.vf v8, v8, ft0
vsetvli x28, x0, e64, m8, tu, mu
vfsgnjx.vf v8, v8, ft0

vsetvli x28, x0, e16, mf2, tu, mu
vfsgnjx.vv v8, v8, v8
vsetvli x28, x0, e16, mf4, tu, mu
vfsgnjx.vv v8, v8, v8
vsetvli x28, x0, e16, m1, tu, mu
vfsgnjx.vv v8, v8, v8
vsetvli x28, x0, e16, m2, tu, mu
vfsgnjx.vv v8, v8, v8
vsetvli x28, x0, e16, m4, tu, mu
vfsgnjx.vv v8, v8, v8
vsetvli x28, x0, e16, m8, tu, mu
vfsgnjx.vv v8, v8, v8
vsetvli x28, x0, e32, mf2, tu, mu
vfsgnjx.vv v8, v8, v8
vsetvli x28, x0, e32, m1, tu, mu
vfsgnjx.vv v8, v8, v8
vsetvli x28, x0, e32, m2, tu, mu
vfsgnjx.vv v8, v8, v8
vsetvli x28, x0, e32, m4, tu, mu
vfsgnjx.vv v8, v8, v8
vsetvli x28, x0, e32, m8, tu, mu
vfsgnjx.vv v8, v8, v8
vsetvli x28, x0, e64, m1, tu, mu
vfsgnjx.vv v8, v8, v8
vsetvli x28, x0, e64, m2, tu, mu
vfsgnjx.vv v8, v8, v8
vsetvli x28, x0, e64, m4, tu, mu
vfsgnjx.vv v8, v8, v8
vsetvli x28, x0, e64, m8, tu, mu
vfsgnjx.vv v8, v8, v8

vsetvli x28, x0, e16, mf2, tu, mu
vfwadd.vf v8, v16, ft0
vsetvli x28, x0, e16, mf4, tu, mu
vfwadd.vf v8, v16, ft0
vsetvli x28, x0, e16, m1, tu, mu
vfwadd.vf v8, v16, ft0
vsetvli x28, x0, e16, m2, tu, mu
vfwadd.vf v8, v16, ft0
vsetvli x28, x0, e16, m4, tu, mu
vfwadd.vf v8, v16, ft0
vsetvli x28, x0, e32, mf2, tu, mu
vfwadd.vf v8, v16, ft0
vsetvli x28, x0, e32, m1, tu, mu
vfwadd.vf v8, v16, ft0
vsetvli x28, x0, e32, m2, tu, mu
vfwadd.vf v8, v16, ft0
vsetvli x28, x0, e32, m4, tu, mu
vfwadd.vf v8, v16, ft0

vsetvli x28, x0, e16, mf2, tu, mu
vfwadd.vv v8, v16, v24
vsetvli x28, x0, e16, mf4, tu, mu
vfwadd.vv v8, v16, v24
vsetvli x28, x0, e16, m1, tu, mu
vfwadd.vv v8, v16, v24
vsetvli x28, x0, e16, m2, tu, mu
vfwadd.vv v8, v16, v24
vsetvli x28, x0, e16, m4, tu, mu
vfwadd.vv v8, v16, v24
vsetvli x28, x0, e32, mf2, tu, mu
vfwadd.vv v8, v16, v24
vsetvli x28, x0, e32, m1, tu, mu
vfwadd.vv v8, v16, v24
vsetvli x28, x0, e32, m2, tu, mu
vfwadd.vv v8, v16, v24
vsetvli x28, x0, e32, m4, tu, mu
vfwadd.vv v8, v16, v24

vsetvli x28, x0, e16, mf2, tu, mu
vfwadd.wf v8, v16, ft0
vsetvli x28, x0, e16, mf4, tu, mu
vfwadd.wf v8, v16, ft0
vsetvli x28, x0, e16, m1, tu, mu
vfwadd.wf v8, v16, ft0
vsetvli x28, x0, e16, m2, tu, mu
vfwadd.wf v8, v16, ft0
vsetvli x28, x0, e16, m4, tu, mu
vfwadd.wf v8, v16, ft0
vsetvli x28, x0, e32, mf2, tu, mu
vfwadd.wf v8, v16, ft0
vsetvli x28, x0, e32, m1, tu, mu
vfwadd.wf v8, v16, ft0
vsetvli x28, x0, e32, m2, tu, mu
vfwadd.wf v8, v16, ft0
vsetvli x28, x0, e32, m4, tu, mu
vfwadd.wf v8, v16, ft0

vsetvli x28, x0, e16, mf2, tu, mu
vfwadd.wv v8, v16, v24
vsetvli x28, x0, e16, mf4, tu, mu
vfwadd.wv v8, v16, v24
vsetvli x28, x0, e16, m1, tu, mu
vfwadd.wv v8, v16, v24
vsetvli x28, x0, e16, m2, tu, mu
vfwadd.wv v8, v16, v24
vsetvli x28, x0, e16, m4, tu, mu
vfwadd.wv v8, v16, v24
vsetvli x28, x0, e32, mf2, tu, mu
vfwadd.wv v8, v16, v24
vsetvli x28, x0, e32, m1, tu, mu
vfwadd.wv v8, v16, v24
vsetvli x28, x0, e32, m2, tu, mu
vfwadd.wv v8, v16, v24
vsetvli x28, x0, e32, m4, tu, mu
vfwadd.wv v8, v16, v24

vsetvli x28, x0, e16, mf2, tu, mu
vfwmul.vf v8, v16, ft0
vsetvli x28, x0, e16, mf4, tu, mu
vfwmul.vf v8, v16, ft0
vsetvli x28, x0, e16, m1, tu, mu
vfwmul.vf v8, v16, ft0
vsetvli x28, x0, e16, m2, tu, mu
vfwmul.vf v8, v16, ft0
vsetvli x28, x0, e16, m4, tu, mu
vfwmul.vf v8, v16, ft0
vsetvli x28, x0, e32, mf2, tu, mu
vfwmul.vf v8, v16, ft0
vsetvli x28, x0, e32, m1, tu, mu
vfwmul.vf v8, v16, ft0
vsetvli x28, x0, e32, m2, tu, mu
vfwmul.vf v8, v16, ft0
vsetvli x28, x0, e32, m4, tu, mu
vfwmul.vf v8, v16, ft0

vsetvli x28, x0, e16, mf2, tu, mu
vfwmul.vv v8, v16, v24
vsetvli x28, x0, e16, mf4, tu, mu
vfwmul.vv v8, v16, v24
vsetvli x28, x0, e16, m1, tu, mu
vfwmul.vv v8, v16, v24
vsetvli x28, x0, e16, m2, tu, mu
vfwmul.vv v8, v16, v24
vsetvli x28, x0, e16, m4, tu, mu
vfwmul.vv v8, v16, v24
vsetvli x28, x0, e32, mf2, tu, mu
vfwmul.vv v8, v16, v24
vsetvli x28, x0, e32, m1, tu, mu
vfwmul.vv v8, v16, v24
vsetvli x28, x0, e32, m2, tu, mu
vfwmul.vv v8, v16, v24
vsetvli x28, x0, e32, m4, tu, mu
vfwmul.vv v8, v16, v24

vsetvli x28, x0, e16, mf2, tu, mu
vfwsub.vf v8, v16, ft0
vsetvli x28, x0, e16, mf4, tu, mu
vfwsub.vf v8, v16, ft0
vsetvli x28, x0, e16, m1, tu, mu
vfwsub.vf v8, v16, ft0
vsetvli x28, x0, e16, m2, tu, mu
vfwsub.vf v8, v16, ft0
vsetvli x28, x0, e16, m4, tu, mu
vfwsub.vf v8, v16, ft0
vsetvli x28, x0, e32, mf2, tu, mu
vfwsub.vf v8, v16, ft0
vsetvli x28, x0, e32, m1, tu, mu
vfwsub.vf v8, v16, ft0
vsetvli x28, x0, e32, m2, tu, mu
vfwsub.vf v8, v16, ft0
vsetvli x28, x0, e32, m4, tu, mu
vfwsub.vf v8, v16, ft0

vsetvli x28, x0, e16, mf2, tu, mu
vfwsub.vv v8, v16, v24
vsetvli x28, x0, e16, mf4, tu, mu
vfwsub.vv v8, v16, v24
vsetvli x28, x0, e16, m1, tu, mu
vfwsub.vv v8, v16, v24
vsetvli x28, x0, e16, m2, tu, mu
vfwsub.vv v8, v16, v24
vsetvli x28, x0, e16, m4, tu, mu
vfwsub.vv v8, v16, v24
vsetvli x28, x0, e32, mf2, tu, mu
vfwsub.vv v8, v16, v24
vsetvli x28, x0, e32, m1, tu, mu
vfwsub.vv v8, v16, v24
vsetvli x28, x0, e32, m2, tu, mu
vfwsub.vv v8, v16, v24
vsetvli x28, x0, e32, m4, tu, mu
vfwsub.vv v8, v16, v24

vsetvli x28, x0, e16, mf2, tu, mu
vfwsub.wf v8, v16, ft0
vsetvli x28, x0, e16, mf4, tu, mu
vfwsub.wf v8, v16, ft0
vsetvli x28, x0, e16, m1, tu, mu
vfwsub.wf v8, v16, ft0
vsetvli x28, x0, e16, m2, tu, mu
vfwsub.wf v8, v16, ft0
vsetvli x28, x0, e16, m4, tu, mu
vfwsub.wf v8, v16, ft0
vsetvli x28, x0, e32, mf2, tu, mu
vfwsub.wf v8, v16, ft0
vsetvli x28, x0, e32, m1, tu, mu
vfwsub.wf v8, v16, ft0
vsetvli x28, x0, e32, m2, tu, mu
vfwsub.wf v8, v16, ft0
vsetvli x28, x0, e32, m4, tu, mu
vfwsub.wf v8, v16, ft0

vsetvli x28, x0, e16, mf2, tu, mu
vfwsub.wv v8, v16, v24
vsetvli x28, x0, e16, mf4, tu, mu
vfwsub.wv v8, v16, v24
vsetvli x28, x0, e16, m1, tu, mu
vfwsub.wv v8, v16, v24
vsetvli x28, x0, e16, m2, tu, mu
vfwsub.wv v8, v16, v24
vsetvli x28, x0, e16, m4, tu, mu
vfwsub.wv v8, v16, v24
vsetvli x28, x0, e32, mf2, tu, mu
vfwsub.wv v8, v16, v24
vsetvli x28, x0, e32, m1, tu, mu
vfwsub.wv v8, v16, v24
vsetvli x28, x0, e32, m2, tu, mu
vfwsub.wv v8, v16, v24
vsetvli x28, x0, e32, m4, tu, mu
vfwsub.wv v8, v16, v24
