# replace_text.cmake
# Variables INPUT_FILE and OUTPUT_FILE must be passed via -D

file(READ "${INPUT_FILE}" CONTENT)

# Perform replacement
string(REPLACE "${MLIR_PYTHON_PACKAGE_PREFIX}._mlir_libs._mlir.ir" "${MLIR_PYTHON_PACKAGE_PREFIX}.ir" MODIFIED_CONTENT "${CONTENT}")

file(WRITE "${OUTPUT_FILE}" "${MODIFIED_CONTENT}")
