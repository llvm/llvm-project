/**
 * @file tpm2_init.c
 * @brief TPM2 initialization and cleanup
 *
 * Uses DSSSL (DSMIL-Grade OpenSSL) for enhanced security:
 * - Post-quantum cryptography support (ML-KEM, ML-DSA)
 * - TPM 2.0 hardware integration
 * - Constant-time operation guarantees
 * - Enhanced side-channel resistance
 *
 * Falls back to OpenSSL 3.x if DSSSL is not available.
 *
 * Classification: UNCLASSIFIED // FOR OFFICIAL USE ONLY
 */

#include "tpm2_compat_accelerated.h"
#include <openssl/evp.h>
#include <openssl/err.h>
#include <string.h>

/* DSSSL-specific headers (OpenSSL 3.x provider API) */
#ifdef HAVE_DSSSL
#include <openssl/provider.h>
#endif

static bool tpm2_initialized = false;
static tpm2_acceleration_flags_t enabled_accel_flags = TPM2_ACCEL_NONE;
static tpm2_security_level_t min_sec_level = TPM2_SEC_STANDARD;

#ifdef HAVE_DSSSL
static OSSL_PROVIDER *dsssl_provider = NULL;
static OSSL_PROVIDER *tpm_provider = NULL;
#endif

tpm2_rc_t tpm2_crypto_init(
    tpm2_acceleration_flags_t accel_flags,
    tpm2_security_level_t min_security_level)
{
    if (tpm2_initialized) {
        return TPM2_RC_SUCCESS;  /* Already initialized */
    }

#ifdef HAVE_DSSSL
    /* Initialize DSSSL providers for PQC and TPM support */
    dsssl_provider = OSSL_PROVIDER_load(NULL, "dsmil");
    if (dsssl_provider != NULL) {
        /* DSSSL provider loaded - enables ML-KEM, ML-DSA, TPM integration */
        tpm_provider = OSSL_PROVIDER_load(NULL, "tpm2");
    } else {
        /* Fallback to default provider */
        OSSL_PROVIDER_load(NULL, "default");
    }
#else
    /* Standard OpenSSL initialization (legacy API for OpenSSL 1.x) */
    OpenSSL_add_all_algorithms();
    ERR_load_crypto_strings();
#endif

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

#ifdef HAVE_DSSSL
    /* Unload DSSSL providers */
    if (tpm_provider != NULL) {
        OSSL_PROVIDER_unload(tpm_provider);
        tpm_provider = NULL;
    }
    if (dsssl_provider != NULL) {
        OSSL_PROVIDER_unload(dsssl_provider);
        dsssl_provider = NULL;
    }
#else
    /* Legacy OpenSSL cleanup (deprecated in OpenSSL 3.x) */
    EVP_cleanup();
    ERR_free_strings();
#endif

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
