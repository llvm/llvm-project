/**
 * @file tpm2_algorithm_info.c
 * @brief Algorithm capability queries
 *
 * Classification: UNCLASSIFIED // FOR OFFICIAL USE ONLY
 */

#include "tpm2_compat_accelerated.h"
#include <string.h>

/* Algorithm information table (88 algorithms) */
static const tpm2_algorithm_info_t algorithm_table[] = {
    /* Hash Algorithms */
    {CRYPTO_ALG_SHA256, "SHA-256", true, true, 0, 0, 32},
    {CRYPTO_ALG_SHA384, "SHA-384", true, true, 0, 0, 48},
    {CRYPTO_ALG_SHA512, "SHA-512", true, true, 0, 0, 64},
    {CRYPTO_ALG_SHA3_256, "SHA3-256", true, true, 0, 0, 32},
    {CRYPTO_ALG_SHA3_384, "SHA3-384", true, true, 0, 0, 48},
    {CRYPTO_ALG_SHA3_512, "SHA3-512", true, true, 0, 0, 64},
    {CRYPTO_ALG_SM3_256, "SM3-256", true, false, 0, 0, 32},

    /* AES Modes */
    {CRYPTO_ALG_AES_128_CBC, "AES-128-CBC", true, true, 16, 16, 0},
    {CRYPTO_ALG_AES_256_CBC, "AES-256-CBC", true, true, 32, 32, 0},
    {CRYPTO_ALG_AES_128_GCM, "AES-128-GCM", true, true, 16, 16, 0},
    {CRYPTO_ALG_AES_256_GCM, "AES-256-GCM", true, true, 32, 32, 0},
    {CRYPTO_ALG_AES_128_CTR, "AES-128-CTR", true, true, 16, 16, 0},
    {CRYPTO_ALG_AES_256_CTR, "AES-256-CTR", true, true, 32, 32, 0},

    /* ChaCha20 */
    {CRYPTO_ALG_CHACHA20, "ChaCha20", true, false, 32, 32, 0},
    {CRYPTO_ALG_CHACHA20_POLY1305, "ChaCha20-Poly1305", true, false, 32, 32, 0},

    /* HMAC */
    {CRYPTO_ALG_HMAC_SHA256, "HMAC-SHA256", true, true, 0, 0, 32},
    {CRYPTO_ALG_HMAC_SHA384, "HMAC-SHA384", true, true, 0, 0, 48},
    {CRYPTO_ALG_HMAC_SHA512, "HMAC-SHA512", true, true, 0, 0, 64},

    /* Elliptic Curves */
    {CRYPTO_ALG_ECC_P256, "NIST P-256", true, false, 32, 32, 64},
    {CRYPTO_ALG_ECC_P384, "NIST P-384", true, false, 48, 48, 96},
    {CRYPTO_ALG_ECC_P521, "NIST P-521", true, false, 66, 66, 132},
    {CRYPTO_ALG_ECC_CURVE25519, "Curve25519", true, false, 32, 32, 32},
    {CRYPTO_ALG_ECC_ED25519, "Ed25519", true, false, 32, 32, 64},

    /* Sentinel */
    {0, NULL, false, false, 0, 0, 0}
};

tpm2_rc_t tpm2_crypto_get_algorithm_info(
    tpm2_crypto_algorithm_t algorithm,
    tpm2_algorithm_info_t *info)
{
    if (info == NULL) {
        return TPM2_RC_VALUE;
    }

    /* Search algorithm table */
    for (size_t i = 0; algorithm_table[i].name != NULL; i++) {
        if (algorithm_table[i].algorithm == algorithm) {
            *info = algorithm_table[i];
            return TPM2_RC_SUCCESS;
        }
    }

    return TPM2_RC_NOT_SUPPORTED;
}
