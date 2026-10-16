#ifdef __cplusplus
extern "C" {
#endif

#ifdef STATIC_RUNTIME
#define TRAP_LINKAGE static __attribute__((noinline, used))
#else
#define TRAP_LINKAGE __attribute__((visibility("hidden"), noinline, used))
#endif

#ifndef CALL_RUNTIME_NAME
#define CALL_RUNTIME_NAME call_main_trap
#endif

#ifdef WITH_STRING_TRAP
TRAP_LINKAGE void __bounds_safety_soft_trap_s(const char *reason) {}
void CALL_RUNTIME_NAME(void) { __bounds_safety_soft_trap_s("fixture"); }
#else
TRAP_LINKAGE void __bounds_safety_soft_trap(void) {}
void CALL_RUNTIME_NAME(void) { __bounds_safety_soft_trap(); }
#endif

#ifdef __cplusplus
}
#endif
