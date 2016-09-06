#!/usr/bin/env python

'''Prepare a code coverage artifact.

- Collate raw profiles into one indexed profile.
- Delete the raw profiles.
- Generate html reports for llvm binaries.
'''

import argparse
import glob
import os
import subprocess
import sys

def merge_raw_profiles(host_llvm_profdata, profile_data_dir, preserve_profiles):
    print ':: Merging raw profiles...',
    sys.stdout.flush()
    raw_profiles = glob.glob(os.path.join(profile_data_dir, '*.profraw'))
    manifest_path = os.path.join(profile_data_dir, 'profiles.manifest')
    profdata_path = os.path.join(profile_data_dir, 'Coverage.profdata')
    with open(manifest_path, 'w') as manifest:
        manifest.write('\n'.join(raw_profiles))
    subprocess.check_call([host_llvm_profdata, 'merge', '-sparse', '-f',
                           manifest_path, '-o', profdata_path])
    if not preserve_profiles:
        for raw_profile in raw_profiles:
            os.remove(raw_profile)
    os.remove(manifest_path)
    print 'Done!'

def prepare_html_report(host_llvm_cov, profile_data_dir, report_dir, binary):
    print ':: Preparing html report for {0}...'.format(binary),
    sys.stdout.flush()
    binary_report_dir = os.path.join(report_dir, os.path.basename(binary))
    profile = os.path.join(profile_data_dir, 'Coverage.profdata')
    subprocess.check_call([host_llvm_cov, 'show', binary, '-format', 'html',
                           '-instr-profile', profile, '-o', binary_report_dir,
                           '-show-line-counts-or-regions',
                           '-Xdemangler', 'c++filt', '-Xdemangler', '-n'])
    with open(os.path.join(binary_report_dir, 'summary.txt'), 'wb') as Summary:
        subprocess.check_call([host_llvm_cov, 'report', binary,
                               '-instr-profile', profile], stdout=Summary)
    print 'Done!'

def prepare_html_reports(host_llvm_cov, profile_data_dir, report_dir, binaries):
    for binary in binaries:
        prepare_html_report(host_llvm_cov, profile_data_dir, report_dir, binary)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('host_llvm_profdata', help='Path to llvm-profdata')
    parser.add_argument('host_llvm_cov', help='Path to llvm-cov')
    parser.add_argument('profile_data_dir',
                       help='Path to the directory containing the raw profiles')
    parser.add_argument('report_dir',
                       help='Path to the output directory for html reports')
    parser.add_argument('binaries', metavar='B', type=str, nargs='+',
                       help='Path to an instrumented binary')
    parser.add_argument('--preserve-profiles',
                       help='Do not delete raw profiles', action='store_true')
    args = parser.parse_args()

    merge_raw_profiles(args.host_llvm_profdata, args.profile_data_dir,
                       args.preserve_profiles)
    prepare_html_reports(args.host_llvm_cov, args.profile_data_dir,
                         args.report_dir, args.binaries)
