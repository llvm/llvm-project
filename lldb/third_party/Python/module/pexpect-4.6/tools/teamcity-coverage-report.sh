#!/bin/bash
# This is to be executed by each individual OS test. It only
# combines coverage files and reports locally to the given
# TeamCity build configuration.
set -e
set -o pipefail
[ -z ${TEMP} ] && TEMP=/tmp

# combine all .coverage* files,
coverage combine

# create ascii report,
report_file=$(mktemp $TEMP/coverage.XXXXX)
coverage report --rcfile=`dirname $0`/../.coveragerc > "${report_file}" 2>/dev/null

# Report Code Coverage for TeamCity, using 'Service Messages',
# https://confluence.jetbrains.com/display/TCD8/How+To...#HowTo...-ImportcoverageresultsinTeamCity
# https://confluence.jetbrains.com/display/TCD8/Custom+Chart#CustomChart-DefaultStatisticsValuesProvidedbyTeamCity
total_no_lines=$(awk '/TOTAL/{printf("%s",$2)}' < "${report_file}")
total_no_misses=$(awk '/TOTAL/{printf("%s",$3)}' < "${report_file}")
total_no_covered=$((${total_no_lines} - ${total_no_misses}))
echo "##teamcity[buildStatisticValue key='CodeCoverageAbsLTotal' value='""${total_no_lines}""']"
echo "##teamcity[buildStatisticValue key='CodeCoverageAbsLCovered' value='""${total_no_covered}""']"

# Display for human consumption and remove ascii file.
cat "${report_file}"
rm "${report_file}"
