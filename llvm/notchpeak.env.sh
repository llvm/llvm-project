export MODULEPATH=$MODULEPATH:/scratch/general/vast/u1290058/setup/spack/share/spack/lmod/linux-rocky8-x86_64/gcc/11.2.0
module load cmake/3.26.0.lua
module load gcc/13.2.0-ycht4e6
module load ninja/1.11.1.lua
module load ccache/4.6.1.lua
module load mold/2.1.0-jxbrdya.lua
export CC=`which gcc`
export CXX=`which g++`
export CCACHE_DIR=/scratch/general/vast/u1290058/mlirWorkspace/ccache
export PATH=`git rev-parse --show-toplevel`/llvm/build/bin:$PATH
