/**
 * @file tpm2_init.c
 * @brief TPM2 initialization and cleanup
 *
 * Classification: UNCLASSIFIED // FOR OFFICIAL USE ONLY
 */

#include "tpm2_compat_accelerated.h"
#include <openssl/evp.h>
#include <openssl/err.h>
#include <string.h>

static bool tpm2_initialized = false;
static tpm2_acceleration_flags_t enabled_accel_flags = TPM2_ACCEL_NONE;
static tpm2_security_level_t min_sec_level = TPM2_SEC_STANDARD;

tpm2_rc_t tpm2_crypto_init(
    tpm2_acceleration_flags_t accel_flags,
    tpm2_security_level_t min_security_level)
{
    if (tpm2_initialized) {
        return TPM2_RC_SUCCESS;  /* Already initialized */
    }

    /* Initialize OpenSSL */
    OpenSSL_add_all_algorithms();
    ERR_load_crypto_strings();

    /* Store configuration */
    enabled_accel_flags = accel_flags;
    min_sec_level = min_security_level;

    tpm2_initialized = true;
    return TPM2_RC_SUCCESS;
}

void tpm2_crypto_cleanup(void)
{
    if (!tpm2_initialized) {
        return;
    }

    /* Cleanup OpenSSL */
    EVP_cleanup();
    ERR_free_strings();

    tpm2_initialized = false;
}

bool tpm2_is_initialized(void)
{
    return tpm2_initialized;
}

tpm2_acceleration_flags_t tpm2_get_accel_flags(void)
{
    return enabled_accel_flags;
}

tpm2_security_level_t tpm2_get_min_security_level(void)
{
    return min_sec_level;
}
