# TPM2 Compatibility Layer Build Configuration
# Classification: UNCLASSIFIED // FOR OFFICIAL USE ONLY
# Version: 2.0.0

cmake_minimum_required(VERSION 3.16)
project(tpm2_compat VERSION 2.0.0 LANGUAGES C)

# Configuration options
option(ENABLE_TPM2_COMPAT "Enable TPM2 compatibility layer" ON)
option(ENABLE_HARDWARE_ACCEL "Enable hardware acceleration (NPU, AES-NI, SHA-NI)" ON)
option(ENABLE_POST_QUANTUM "Enable post-quantum cryptography (requires liboqs)" OFF)
option(USE_OPENSSL_FALLBACK "Allow fallback to OpenSSL if DSSSL not found (DSSSL is default)" OFF)
option(BUILD_TPM2_TESTS "Build TPM2 unit tests" OFF)
option(BUILD_TPM2_BENCHMARKS "Build TPM2 performance benchmarks" OFF)

# C Standard
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

# Compiler flags
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(
        -Wall
        -Wextra
        -Wpedantic
        -Werror=implicit-function-declaration
        -Werror=incompatible-pointer-types
        -fstack-protector-strong
        -D_FORTIFY_SOURCE=2
    )

    if(ENABLE_HARDWARE_ACCEL)
        add_compile_options(
            -march=native
            -maes
            -msha
            -mavx2
        )
    endif()
endif()

# Find DSSSL (DSMIL-Grade OpenSSL fork with PQC support)
# DSSSL is the default and required cryptographic library for DSLLVM
# Use -DUSE_OPENSSL_FALLBACK=ON to allow OpenSSL 3.x fallback
if(USE_OPENSSL_FALLBACK)
    find_package(DSSSL 3.0 QUIET)
    if(NOT DSSSL_FOUND)
        message(WARNING "DSSSL not found, falling back to OpenSSL 3.x")
        message(WARNING "For full DSMIL security features, install DSSSL: https://github.com/SWORDIntel/DSSSL")
        find_package(OpenSSL 3.0 REQUIRED)
        set(CRYPTO_LIB OpenSSL::Crypto)
        set(CRYPTO_NAME "OpenSSL 3.x (fallback)")
        add_compile_definitions(USE_OPENSSL_FALLBACK=1)
    else()
        set(CRYPTO_LIB DSSSL::Crypto)
        set(CRYPTO_NAME "DSSSL")
        add_compile_definitions(HAVE_DSSSL=1)
        message(STATUS "Using DSSSL (DSMIL-Grade OpenSSL) for cryptography")
    endif()
else()
    # DSSSL is REQUIRED by default for DSLLVM/DSMIL projects
    find_package(DSSSL 3.0 REQUIRED)
    set(CRYPTO_LIB DSSSL::Crypto)
    set(CRYPTO_NAME "DSSSL")
    add_compile_definitions(HAVE_DSSSL=1)
    message(STATUS "Using DSSSL (DSMIL-Grade OpenSSL) for cryptography")
endif()

# Optional: liboqs for post-quantum cryptography
if(ENABLE_POST_QUANTUM)
    find_package(liboqs)
    if(liboqs_FOUND)
        add_compile_definitions(HAVE_LIBOQS=1)
    else()
        message(WARNING "liboqs not found - post-quantum crypto will be disabled")
        set(ENABLE_POST_QUANTUM OFF)
    endif()
endif()

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${OPENSSL_INCLUDE_DIR}
)

# Source files
set(TPM2_SOURCES
    src/core/tpm2_init.c
    src/core/tpm2_algorithm_info.c
    src/hash/tpm2_hash.c
    src/symmetric/tpm2_symmetric.c
    src/symmetric/tpm2_aead.c
    src/mac/tpm2_hmac.c
    src/kdf/tpm2_kdf.c
    src/kdf/tpm2_hkdf.c
    src/kdf/tpm2_pbkdf2.c
    src/keyagreement/tpm2_ecdh.c
    src/asymmetric/tpm2_rsa.c
    src/asymmetric/tpm2_ecc.c
    src/signature/tpm2_signature.c
    src/core/tpm2_utils.c
)

# Post-quantum sources (conditional)
if(ENABLE_POST_QUANTUM)
    list(APPEND TPM2_SOURCES
        src/pqc/tpm2_kyber.c
        src/pqc/tpm2_dilithium.c
        src/pqc/tpm2_falcon.c
    )
endif()

# Create TPM2 compatibility library
add_library(tpm2_compat STATIC ${TPM2_SOURCES})

target_include_directories(tpm2_compat PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(tpm2_compat
    PUBLIC
        ${CRYPTO_LIB}
)

if(ENABLE_POST_QUANTUM AND liboqs_FOUND)
    target_link_libraries(tpm2_compat PUBLIC liboqs::oqs)
endif()

# Set library properties
set_target_properties(tpm2_compat PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 2
    PUBLIC_HEADER "include/tpm2_types.h;include/tpm2_compat_accelerated.h"
)

# Installation
install(TARGETS tpm2_compat
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    PUBLIC_HEADER DESTINATION include/tpm2_compat
)

# Tests (if enabled)
if(BUILD_TPM2_TESTS)
    enable_testing()
    add_subdirectory(tests/unit)
endif()

# Benchmarks (if enabled)
if(BUILD_TPM2_BENCHMARKS)
    add_subdirectory(tests/performance)
endif()

# Print configuration summary
message(STATUS "========================================")
message(STATUS "TPM2 Compatibility Layer Configuration:")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  Crypto Library: ${CRYPTO_NAME}")
message(STATUS "  OpenSSL Fallback: ${USE_OPENSSL_FALLBACK}")
message(STATUS "  Hardware Acceleration: ${ENABLE_HARDWARE_ACCEL}")
message(STATUS "  Post-Quantum Crypto: ${ENABLE_POST_QUANTUM}")
message(STATUS "  Build Tests: ${BUILD_TPM2_TESTS}")
message(STATUS "  Build Benchmarks: ${BUILD_TPM2_BENCHMARKS}")
message(STATUS "========================================")
if(NOT USE_OPENSSL_FALLBACK AND NOT DSSSL_FOUND)
    message(FATAL_ERROR "DSSSL is required for DSLLVM. Install from https://github.com/SWORDIntel/DSSSL or use -DUSE_OPENSSL_FALLBACK=ON")
endif()
