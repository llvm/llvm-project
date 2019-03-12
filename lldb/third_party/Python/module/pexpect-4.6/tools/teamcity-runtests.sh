#!/bin/bash
#
# This script assumes that the project 'ptyprocess' is
# available in the parent of the project's folder.
set -e
set -o pipefail

if [ -z $1 ]; then
	echo "$0 (2.6|2.7|3.3|3.4)"
	exit 1
fi

export PYTHONIOENCODING=UTF8
export LANG=en_US.UTF-8

pyversion=$1
shift
here=$(cd `dirname $0`; pwd)
osrel=$(uname -s)
venv=teamcity-pexpect
venv_wrapper=$(which virtualenvwrapper.sh)

if [ -z $venv_wrapper ]; then
	echo "virtualenvwrapper.sh not found in PATH." >&2
	exit 1
fi

. ${venv_wrapper}
rmvirtualenv ${venv} || true
mkvirtualenv -p `which python${pyversion}` ${venv} || true
workon ${venv}

# install ptyprocess
cd $here/../../ptyprocess
pip uninstall --yes ptyprocess || true
python setup.py install

# install all test requirements
pip install --upgrade pytest-cov coverage coveralls pytest-capturelog

# run tests
cd $here/..
ret=0
py.test \
	--cov pexpect \
	--cov-config .coveragerc \
	--junit-xml=results.${osrel}.py${pyversion}.xml \
	--verbose \
	--verbose \
	"$@" || ret=$?

if [ $ret -ne 0 ]; then
	# we always exit 0, preferring instead the jUnit XML
	# results to be the dominate cause of a failed build.
	echo "py.test returned exit code ${ret}." >&2
	echo "the build should detect and report these failing tests." >&2
fi

# combine all coverage to single file, report for this build,
# then move into ./build-output/ as a unique artifact to allow
# the final "Full build" step to combine and report to coveralls.io
`dirname $0`/teamcity-coverage-report.sh
mkdir -p build-output
mv .coverage build-output/.coverage.${osrel}.py{$pyversion}.$RANDOM.$$
