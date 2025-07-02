LLVM_DIR=/home/mlevental/dev_projects/llvm-project/llvm
TARGET=AMDGPU
./bin/llvm-tblgen -unison $LLVM_DIR/lib/Target/$TARGET/$TARGET.td \
            -I $LLVM_DIR/include -I $LLVM_DIR/lib/Target/$TARGET \
            -o $TARGET.yaml
