#include <mach/mach.h>
#include <stdint.h>

// Define the SVE/SME/SME2 thread status structures
// flavors, and sizes so this can build against an
// older SDK which does not have these definitions
// yet.

#if !defined(ARM_SME_STATE)

#define _STRUCT_ARM_SME_STATE struct arm_sme_state
_STRUCT_ARM_SME_STATE {
  uint64_t svcr;
  uint64_t tpidr2_el0;
  uint16_t svl_b;
};

#define _STRUCT_ARM_SVE_Z_STATE struct arm_sve_z_state
_STRUCT_ARM_SVE_Z_STATE { char z[16][256]; }
__attribute__((aligned(alignof(unsigned int))));

#define _STRUCT_ARM_SVE_P_STATE struct arm_sve_p_state
_STRUCT_ARM_SVE_P_STATE { char p[16][256 / 8]; }
__attribute__((aligned(alignof(unsigned int))));

#define _STRUCT_ARM_SME_ZA_STATE struct arm_sme_za_state
_STRUCT_ARM_SME_ZA_STATE { char za[4096]; }
__attribute__((aligned(alignof(unsigned int))));

#define _STRUCT_ARM_SME2_STATE struct arm_sme2_state
_STRUCT_ARM_SME2_STATE { char zt0[64]; }
__attribute__((aligned(alignof(unsigned int))));

#define ARM_SME_STATE 28
#define ARM_SVE_Z_STATE1 29
#define ARM_SVE_Z_STATE2 30
#define ARM_SVE_P_STATE 31
#define ARM_SME_ZA_STATE1 32
#define ARM_SME_ZA_STATE2 33
#define ARM_SME_ZA_STATE3 34
#define ARM_SME_ZA_STATE4 35
#define ARM_SME_ZA_STATE5 36
#define ARM_SME_ZA_STATE6 37
#define ARM_SME_ZA_STATE7 38
#define ARM_SME_ZA_STATE8 39
#define ARM_SME_ZA_STATE9 40
#define ARM_SME_ZA_STATE10 41
#define ARM_SME_ZA_STATE11 42
#define ARM_SME_ZA_STATE12 42
#define ARM_SME_ZA_STATE13 44
#define ARM_SME_ZA_STATE14 45
#define ARM_SME_ZA_STATE15 46
#define ARM_SME_ZA_STATE16 47
#define ARM_SME2_STATE 48

typedef _STRUCT_ARM_SME_STATE arm_sme_state_t;
typedef _STRUCT_ARM_SVE_Z_STATE arm_sve_z_state_t;
typedef _STRUCT_ARM_SVE_P_STATE arm_sve_p_state_t;
typedef _STRUCT_ARM_SME_ZA_STATE arm_sme_za_state_t;
typedef _STRUCT_ARM_SME2_STATE arm_sme2_state_t;

#define ARM_SME_STATE_COUNT                                                    \
  ((mach_msg_type_number_t)(sizeof(arm_sme_state_t) / sizeof(uint32_t)))

#define ARM_SVE_Z_STATE_COUNT                                                  \
  ((mach_msg_type_number_t)(sizeof(arm_sve_z_state_t) / sizeof(uint32_t)))

#define ARM_SVE_P_STATE_COUNT                                                  \
  ((mach_msg_type_number_t)(sizeof(arm_sve_p_state_t) / sizeof(uint32_t)))

#define ARM_SME_ZA_STATE_COUNT                                                 \
  ((mach_msg_type_number_t)(sizeof(arm_sme_za_state_t) / sizeof(uint32_t)))

#define ARM_SME2_STATE_COUNT                                                   \
  ((mach_msg_type_number_t)(sizeof(arm_sme2_state_t) / sizeof(uint32_t)))

#endif // !defined(ARM_SME_STATE)
