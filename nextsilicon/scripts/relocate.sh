#!/bin/bash -eu
# This script handles relocation toolchain install directory
# When moved around from one machine to another or when moved to different
# Folder on same machine

SCRIPT_PATH=$(cd $(dirname $0) && pwd)

NEXT_PATH=$(realpath ${SCRIPT_PATH}/..)

if ! which patchelf > /dev/null ; then
	echo "patchelf not found, skipping binaries patch" >&2
	echo "flang compiler and third party libraries will not be useable" >&2
	exit 0
fi

# HOME_EXECUTABLES are binaries that were built with flatcc
HOME_EXECUTABLES=$(find ${NEXT_PATH}/bin -type f -executable)
HOME_LIBRARIES=$(find ${NEXT_PATH}/lib* -type f -executable)
# LLVM_EXECUTABLES are binaries that were potentially built with flatcc (flang, etc.)
LLVM_EXECUTABLES=$(find ${NEXT_PATH}/llvm/bin -type f -executable)
SYSROOT_EXECUTABLES=$(find ${NEXT_PATH}/sysroot/usr/*bin -type f -executable)
SYSROOT_LIBRARIES=$(find ${NEXT_PATH}/sysroot/usr/lib* -type f)

# Patch path to dynamic linker. Path has to be absolute
for elf in \
	$HOME_EXECUTABLES \
	$HOME_LIBRARIES \
	$LLVM_EXECUTABLES \
	$SYSROOT_EXECUTABLES \
	$SYSROOT_LIBRARIES ; do

	interpreter=$(patchelf --print-interpreter "${elf}" 2>/dev/null || true)

	if [[ ${interpreter} = *"sysroot/usr/lib/ld-linux-x86-64.so.2" ]]; then
		patchelf --set-interpreter "${NEXT_PATH}/sysroot/usr/lib/ld-linux-x86-64.so.2" "${elf}"
	fi

	rpath=$(patchelf --print-rpath "${elf}" 2>/dev/null || true)

	if [[ "${rpath}" != *"sysroot/usr/lib"* && "${rpath}" != "/opt/nextsilicon/lib" ]]; then
		continue
	fi

	newrpath=''

	if [[ "${rpath}" != *'$ORIGIN/../lib'* ]]; then
		newrpath='$ORIGIN/../lib'
	fi

	for p in $(echo "${rpath}" | tr ":" "\n") ; do
		# Some packages install into usr/lib64 instead of usr/lib,
		# and require both to function
		if [[ "${p}" =~ .*(sysroot/usr/lib64.*) ]]; then
			newrpath+=":${NEXT_PATH}/${BASH_REMATCH[1]}"
		elif [[ "${p}" =~ .*(sysroot/usr/lib.*) ]]; then
			newrpath+=":${NEXT_PATH}/${BASH_REMATCH[1]}"
		elif [[ "${p}" =~ /opt/nextsilicon/(lib.*) ]]; then
			newrpath+=":${NEXT_PATH}/${BASH_REMATCH[1]}"
		elif [[ "${p}" =~ /tmp/test/(lib.*) ]]; then
			newrpath+=":${NEXT_PATH}/${BASH_REMATCH[1]}"
		else
			newrpath+=":${p}"
		fi
	done

	patchelf --force-rpath --set-rpath "${newrpath}" "${elf}"
done

for elf in $HOME_EXECUTABLES ; do

	rpath=$(patchelf --print-rpath "${elf}" 2>/dev/null || true)

	if [[ ${rpath} != *"llvm/lib"* && ${rpath} != *"python/lib"* ]]; then
		continue
	fi

	# ensure that all references to */llvm/lib are preserved
	# only necessary for custom prefix packaged installations
	# - notably elrond (not part of toolchain)
	newrpath=''

	for p in $(echo "${rpath}" | tr ":" "\n") ; do
		if [[ "${p}" = *"llvm/lib"* ]]; then
			newrpath+=':$ORIGIN/../llvm/lib'
		elif [[ "${p}" = *"python/lib"* ]]; then
			newrpath+=':$ORIGIN/../python/lib'
		else
			newrpath+=":${p}"
		fi
	done

	if [[ "${rpath}" != *'$ORIGIN/../deps/lib'* ]]; then
		newrpath='$ORIGIN/../deps/lib:'"${newrpath}"
	fi

	if [[ "${rpath}" != *'$ORIGIN/../lib'* ]]; then
		newrpath='$ORIGIN/../lib:'"${newrpath}"
	fi

	patchelf --force-rpath --set-rpath "${newrpath}" "${elf}"
done

find "${NEXT_PATH}/sysroot/usr/lib" -type f -name '*.la' \
	-exec sed -i 's|/[^ ]*/sysroot/usr/lib|'${NEXT_PATH}'/sysroot/usr/lib|g' {} \;
