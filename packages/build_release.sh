#!/bin/bash

set -o pipefail
PKG_NAME="${PKG_NAME:-nextllvm}"
PKG_PATH="package_path"
BRANCH_NAME="${BRANCH_NAME:-next_release_170}"
GIT_SHA=""
MAJOR_VERSION=""
DEPENDENCIES=""
BUILD_ID="${BUILD_ID:-1234}"
NEXT_HOME="${NEXT_HOME:-/opt/nextsilicon}"
PATH_PREFIX="opt/nextsilicon"
INPUT_ARTIFACT="${INPUT_ARTIFACT:-packages/artifacts.txt}"
INPUT_NEXTCRT_SOURCES="${INPUT_NEXTCRT_SOURCES:-packages/nextcrt_sources.txt}"
TAR_EXCLUDES_FILE="${TAR_EXCLUDES_FILE:-packages/sources_excludes.txt}"
TAR_EXCLUDES=""
OS_VARIANT=""
PACK_RELEASE=false
COPY_ARTIFACTS=false
PACK_SOURCES=false
UPDATE_ARTIFACTOY=false
REPO=${K8S_JFROG_URL}

function clean {
  # cleaning artifacts path
  rm -rf "${PKG_PATH}"
  rm -f "${PKG_NAME}"*.rpm "${PKG_NAME}"*.deb "${PKG_NAME}"*.xz
}

function copy_setup {

  input_file=$1
  src_path=$2
  path_prefix=$3
  marker=0

  while IFS= read -r line
  do
    # ignore comment lines (begins with '#')
    [[ "${line}" =~ ^\# ]] && continue
    # ignore empty lines
    [[ "${line}" == '' ]] && continue
    # a hook for deploying files directly from NEXT_HOME path
    orig=$(echo "${line}"| awk -F':' '{print $1}')
    dest=$(echo "${line}"| awk -F':' '{print $2}')
    # if the packaging path doesn't exist, create it
    if [[ ! -d "${PKG_PATH}/${path_prefix}/$(dirname\
 "${dest}")" ]]; then
      mkdir -p "${PKG_PATH}/${path_prefix}/$(dirname\
 "${dest}")"
    fi
    # if the path is a directory
    if [[ -d ${src_path}/${orig} ]]; then
      echo "cp -a ${src_path}/${orig} ${PKG_PATH}/${path_prefix}/${dest}"
      cp -a "${src_path}/${orig}" "${PKG_PATH}/${path_prefix}/${dest}"\
       && continue
    # if the path is a file
    elif [[ -f ${src_path}/${orig} ]]; then
      echo "cp -a ${src_path}/${orig} ${PKG_PATH}/${path_prefix}/${dest}"
      cp -a "${src_path}/${orig}" "${PKG_PATH}/${path_prefix}/${dest}"\
       && continue
      # if the path is other directory
    elif [[ -d ${orig} ]]; then
      echo "cp -a ${orig} ${PKG_PATH}/${path_prefix}"
      cp -a "${orig}" "${PKG_PATH}/${path_prefix}" && continue
      # if the path is other file
    elif [[ -f ${orig} ]]; then
      echo "cp -a ${orig} ${PKG_PATH}/${orig}"
      cp -a "${orig}" "${PKG_PATH}/${orig}" && continue
    else
      echo "Attention - ${orig} is not a regular file."
      marker=$((marker + 1))
    fi
    if [[ ! -e ${src_path}/${orig} ]]; then
      echo "Attention - ${line} doesn't exist under ${src_path}"
      marker=$((marker + 1))
      continue
    fi
  done < "${input_file}"
  if [[ $marker -gt 0 ]];then
    exit 1
  fi

}

function set_parameters {

  MAJOR_VERSION="$("${NEXT_HOME}"/bin/nextcc --version | grep "clang version" | awk '{print $4}')"
  if [ "${MAJOR_VERSION}" = "" ]; then
    echo "unable to get clang version."
    exit 1
  fi

  DEB_VERSION="${MAJOR_VERSION}-${BUILD_ID}"
  DEB_NAME="${PKG_NAME}_${DEB_VERSION}_amd64.deb"
  RPM_VERSION="${MAJOR_VERSION}-${BUILD_ID}"
  RPM_NAME="${PKG_NAME}-${RPM_VERSION}.x86_64.rpm"
  GIT_SHA="$(git rev-parse HEAD)"
}

function setup {

  # creating artifacts path
  mkdir -p "${PKG_PATH}/${PATH_PREFIX}"

  echo "${INPUT_ARTIFACT}: copying build artifacts files to ${PKG_PATH}/${PATH_PREFIX}"
  # copying files needed for packaging process
  copy_setup "${INPUT_ARTIFACT}" "${NEXT_HOME}" "${PATH_PREFIX}"

}


function build_deb {

  echo "building deb package $DEB_NAME"

  # creating debian path
  mkdir "${PKG_PATH}/DEBIAN"

  # copying debian control file
  cp packages/control "${PKG_PATH}/DEBIAN/"
  # copy & setup postinst file if exists
  if [ -f packages/postinst ]; then
    cp -a packages/postinst "${PKG_PATH}/DEBIAN/"
    chmod 755 "${PKG_PATH}/DEBIAN/postinst"
  fi

  # replacing macros @control file
  sed -i "s/{{VERSION}}/${DEB_VERSION}/" "${PKG_PATH}"/DEBIAN/control || exit 1
  sed -i "s/{{BRANCH}}/${BRANCH_NAME}/" "${PKG_PATH}"/DEBIAN/control || exit 1
  sed -i "s/{{GIT_SHA}}/${GIT_SHA}/" "${PKG_PATH}"/DEBIAN/control || exit 1
  sed -i "s/{{PKG_NAME}}/${PKG_NAME}/" "${PKG_PATH}"/DEBIAN/control || exit 1

  # build deb package and list its metadata
  echo "dpkg-deb --root-owner-group -v -D --build ${PKG_PATH} ${DEB_NAME}"
  dpkg-deb -v -D --root-owner-group --build "${PKG_PATH}" "${DEB_NAME}" || exit 1
  dpkg-deb -I "${DEB_NAME}"
}

function build_rpm {

  echo "building rpm package $RPM_NAME"

  SPEC_FILE="${PKG_PATH}/SPECS/${PKG_NAME}.spec"
  BUILDROOT="${PKG_PATH}/BUILDROOT/${PKG_NAME}-${RPM_VERSION}.x86_64"
  CWD=$(pwd)

  # creating rpm paths
  mkdir -p "${PKG_PATH}"/{RPMS,SPECS,BUILDROOT}
  mkdir -p "${PKG_PATH}/RPMS/x86_64"
  mkdir "${BUILDROOT}"

  mv "${PKG_PATH}/opt" "${BUILDROOT}"

  # copying rpm spec file
  cp "packages/${PKG_NAME}.spec" "${SPEC_FILE}"

  # replacing macros @spec file
  if [[ "${OS_VARIANT}" =~ ^opensuse ]]; then
    DEPENDENCIES="libatomic1, mpfr-devel, gmp-devel"
  else
    DEPENDENCIES="libatomic, mpfr, gmp"
  fi
  sed -i "s/{{MAJOR_VERSION}}/${MAJOR_VERSION}/" "${SPEC_FILE}"
  sed -i "s/{{RELEASE}}/${BUILD_ID}/" "${SPEC_FILE}"
  sed -i "s/{{BRANCH}}/${BRANCH_NAME}/" "${SPEC_FILE}"
  sed -i "s/{{GIT_SHA}}/${GIT_SHA}/" "${SPEC_FILE}"
  sed -i "s/{{PKG_NAME}}/${PKG_NAME}/" "${SPEC_FILE}"
  sed -i "s/{{DEPENDENCIES}}/${DEPENDENCIES}/" "${SPEC_FILE}"

  # build rpm and delete artifacts when done
  echo "rpmbuild -bb --target x86_64 --define \"_topdir ${CWD}/${PKG_PATH}\" --define \"NextSilicon\" ${SPEC_FILE}"
  rpmbuild -bb --target x86_64 --define "_topdir ${CWD}/${PKG_PATH}"  \
           --define "_gpg_name NextSilicon" "${SPEC_FILE}" || exit 1
  mv "${PKG_PATH}/RPMS/x86_64/${RPM_NAME}" .
  rpm -qip "${RPM_NAME}"
}

function gen_tar_exclude {

  exclude_file=${1}
  while IFS= read -r line; do
    # ignore comment lines (begins with '#')
    [[ "${line}" =~ ^\# ]] && continue
    # ignore empty lines
    [[ "${line}" == '' ]] && continue

    TAR_EXCLUDES="${TAR_EXCLUDES} --exclude=${line%/}"
  done < "${exclude_file}"

}

function pack_conan {

  echo "conan remove \"*\" -s -b -f"
  conan remove "*" -s -b -f || exit 1
  echo "Backing .conan"
  echo "tar -cJf ${PKG_NAME}-${RPM_VERSION}-conan.xz -C /conan/.conan ."
  tar -cJf "${PKG_NAME}-${RPM_VERSION}-conan.tar.xz" -C /conan/.conan .

}


function sources_build_metadata {

cat > "${PKG_NAME}_info.txt" <<- EOF
Package:  ${PKG_NAME}
Version:  ${RPM_VERSION}
Branch:   ${BRANCH_NAME}
GIT_SHA:  ${GIT_SHA}
Homepage: www.nextsilicon.com
Support:  <support@nextsilicon.com>
EOF

}

function build_sources {

  pack_conan

  # exclude untracked files generated by the build and .gitignore
  for line in $(git ls-files --others --directory); do
    TAR_EXCLUDES="${TAR_EXCLUDES} --exclude=${line%/}"
  done
  # backing partial .git folder
  mkdir .git_tar
  cp -a .git/{HEAD,objects,refs} .git_tar/
  rm -rf .git_tar/objects/* .git_tar/refs/remotes/ .git_tar/refs/tags/
  rm -rf .git_backup
  mv .git/ .git_backup/
  mv .git_tar/ .git/
  sed -i '/.git$/d' "${TAR_EXCLUDES_FILE}"
  gen_tar_exclude "${TAR_EXCLUDES_FILE}"

  sources_build_metadata

  cd ..
  cmd="tar ${TAR_EXCLUDES} --exclude=.git_backup -cJf \
          ${PKG_NAME}-${RPM_VERSION}-sources.xz next-llvm-project"
  echo "Building sources"
  echo "${cmd}"
  ${cmd}
  cp "${PKG_NAME}-${RPM_VERSION}-sources.xz" next-llvm-project/
  cd next-llvm-project/ || exit
  rm -rf .git/
  mv .git_backup/ .git/
}

function update_artifactory {

  echo "Updating remote artifactory ${REPO}"

  # Upload deb packages to remote repository
  pkgs=$(find . -type f -name "${PKG_NAME}*.deb" 2> /dev/null | wc -l)
  if [ "${pkgs}" -gt 0 ]; then
    REMOTE_REPO="${REPO}/generic-repo/nextsilicon-files/nextllvm/${OS_VARIANT}/${BRANCH_NAME}/"
    find . -type f -name "${PKG_NAME}*.deb" | while read -r pkg
    do
      echo "Uploading ${pkg} to ${REMOTE_REPO}"
      curl -Sf -u ${USERNAME}:${PASSWORD} -X PUT "${REMOTE_REPO}" -T ${pkg}
    done
  elif [[ "${OS_VARIANT}" =~ ^debian ]]; then
    echo "Missing ${OS_VARIANT} packages"
    exit 1
  fi

  # Upload rpm packages to remote repository
  pkgs=$(find . -type f -name "${PKG_NAME}*.rpm" 2> /dev/null | wc -l)
  if [ "${pkgs}" -gt 0 ]; then
    REMOTE_REPO="${REPO}/generic-repo/nextsilicon-files/nextllvm/${OS_VARIANT}/${BRANCH_NAME}/"
    find . -type f -name "${PKG_NAME}*.rpm" | while read -r pkg
    do
      echo "Uploading ${pkg} to ${REMOTE_REPO}"
      curl -Sf -u ${USERNAME}:${PASSWORD} -X PUT "${REMOTE_REPO}" -T ${pkg}
    done
  elif [[ "${OS_VARIANT}" =~ ^centos ]] || [[ "${OS_VARIANT}" =~ ^rhel ]] || [[ "${OS_VARIANT}" =~ ^rocky ]]; then
    echo "Missing ${OS_VARIANT} packages"
    exit 1
  fi

}


function usage() {
    echo "Build next-llvm-project OS variant packages"
    echo "Usage: $0 [OPTIONS]"
    echo "  --os-variant             OS variant: options are centos-XX, debian-XX, opensuse-XX \
rocky-8, rhel-8"
    echo "  --pack-release           Pack release package"
    echo "  --pack-sources           Pack sources xz"
    echo "  --update-artifactory     Upload packages to artifactory artifactory"
    echo "  --repo-user              Set repository user name"
    echo "  --repo-password          Set repository password"
    echo "  --copy-artifacts         Pack release sources"
}

# main
while [[ $# -gt 0 ]] ; do
    key="$1"
    shift
    case $key in
        -h|--help)
            usage
            exit 0
        ;;
        --os-variant)
            OS_VARIANT="${1}"
            if [[ ! "${OS_VARIANT}" =~ ^(centos|debian|opensuse|rhel|rocky) ]]; then
              echo "--os-variant: options are centos-XX, debian-XX, rhel-X, rocky-X or opensuse-XX,\
got \"${OS_VARIANT}\""
              usage
              exit 1
            fi
            shift
        ;;
        --pack-release)
            PACK_RELEASE=true
        ;;
        --pack-sources)
            PACK_SOURCES=true
        ;;
        --copy-artifacts)
            COPY_ARTIFACTS=true
        ;;
        --update-artifactory)
          UPDATE_ARTIFACTOY=true
        ;;
        --repo-user)
            USERNAME="${1}"
            shift
        ;;
        --repo-password)
            PASSWORD="${1}"
            shift
        ;;
        *)
            # unknown option
            echo "Unknown parameter $key" >&2
            exit 1
        ;;
    esac
done

if [ "${OS_VARIANT}" = "" ]; then
  usage
  exit 1
fi

if [ "${PACK_RELEASE}" = true ]; then
  set_parameters

fi

if [ "${COPY_ARTIFACTS}" = true ]; then
  clean

  setup
fi

if [ "${PACK_RELEASE}" = true ]; then

  clean

  setup

  if [[ "${OS_VARIANT}" =~ ^debian ]]; then
    build_deb
  else
    build_rpm
  fi
fi

if [ ${PACK_SOURCES} = true ]; then
  build_sources
fi

if [ "${UPDATE_ARTIFACTOY}" = true ]; then
  update_artifactory
fi

