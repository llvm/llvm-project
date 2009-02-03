#!/usr/bin/perl
#
# Program:  find-cycles.pl
#
# Synopsis: Given a list of possibly cyclic dependencies, merge all the
#           cycles.  This makes it possible to topologically sort the
#           dependencies between different parts of LLVM.
#
# Syntax:   find-cycles.pl < LibDeps.txt > FinalLibDeps.txt
#
# Input:    cycmem1: cycmem2 dep1 dep2
#           cycmem2: cycmem1 dep3 dep4
#           boring: dep4
#
# Output:   cycmem1 cycmem2: dep1 dep2 dep3 dep4
#           boring: dep4
#
# This file was written by Eric Kidd, and is placed into the public domain.
#

use 5.006;
use strict;
use warnings;

my %DEPS;
my @CYCLES;
sub find_all_cycles;

# Read our dependency information.
while (<>) {
    chomp;
    my ($module, $dependency_str) = /^\s*([^:]+):\s*(.*)\s*$/;
    die "Malformed data: $_" unless defined $dependency_str;
    my @dependencies = split(/ /, $dependency_str);
    $DEPS{$module} = \@dependencies;
}

# Partition our raw dependencies into sets of cyclically-connected nodes.
find_all_cycles();

# Print out the finished cycles, with their dependencies.
my @output;
my $cycles_found = 0;
foreach my $cycle (@CYCLES) {
    my @modules = sort keys %{$cycle};

    # Merge the dependencies of all modules in this cycle.
    my %dependencies;
    foreach my $module (@modules) {
        @dependencies{@{$DEPS{$module}}} = 1;
    }

    # Prune the known cyclic dependencies.
    foreach my $module (@modules) {
        delete $dependencies{$module};
    }

    # Warn about possible linker problems.
    my @archives = grep(/\.a$/, @modules);
    if (@archives > 1) {
        $cycles_found = $cycles_found + 1;
        print STDERR "find-cycles.pl: Circular dependency between *.a files:\n";
        print STDERR "find-cycles.pl:   ", join(' ', @archives), "\n";
        push @modules, @archives; # WORKAROUND: Duplicate *.a files. Ick.
    }

    # Add to our output.  (@modules is already as sorted as we need it to be.)
    push @output, (join(' ', @modules) . ': ' .
                   join(' ', sort keys %dependencies) . "\n");
}
print sort @output;

exit $cycles_found;

#==========================================================================
#  Depedency Cycle Support
#==========================================================================
#  For now, we have cycles in our dependency graph.  Ideally, each cycle
#  would be collapsed down to a single *.a file, saving us all this work.
#
#  To understand this code, you'll need a working knowledge of Perl 5,
#  and possibly some quality time with 'man perlref'.

my %SEEN;
my %CYCLES;
sub find_cycles ($@);
sub found_cycles ($@);

sub find_all_cycles {
    # Find all multi-item cycles.
    my @modules = sort keys %DEPS;
    foreach my $module (@modules) { find_cycles($module); }

    # Build fake one-item "cycles" for the remaining modules, so we can
    # treat them uniformly.
    foreach my $module (@modules) {
        unless (defined $CYCLES{$module}) {
            my %cycle = ($module, 1);
            $CYCLES{$module} = \%cycle;
        }
    }

    # Find all our unique cycles.  We have to do this the hard way because
    # we apparently can't store hash references as hash keys without making
    # 'strict refs' sad.
    my %seen;
    foreach my $cycle (values %CYCLES) {
        unless ($seen{$cycle}) {
            $seen{$cycle} = 1;
            push @CYCLES, $cycle;
        }
    }
}

# Walk through our graph depth-first (keeping a trail in @path), and report
# any cycles we find.
sub find_cycles ($@) {
    my ($module, @path) = @_;
    if (str_in_list($module, @path)) {
        found_cycle($module, @path);
    } else {
        return if defined $SEEN{$module};
        $SEEN{$module} = 1;
        foreach my $dep (@{$DEPS{$module}}) {
            find_cycles($dep, @path, $module);
        }
    }
}

# Give a cycle, attempt to merge it with pre-existing cycle data.
sub found_cycle ($@) {
    my ($module, @path) = @_;

    # Pop any modules which aren't part of our cycle.
    while ($path[0] ne $module) { shift @path; }
    #print join("->", @path, $module) . "\n";

    # Collect the modules in our cycle into a hash.
    my %cycle;
    foreach my $item (@path) {
        $cycle{$item} = 1;
        if (defined $CYCLES{$item}) {
            # Looks like we intersect with an existing cycle, so merge
            # all those in, too.
            foreach my $old_item (keys %{$CYCLES{$item}}) {
                $cycle{$old_item} = 1;
            }
        }
    }

    # Update our global cycle table.
    my $cycle_ref = \%cycle;
    foreach my $item (keys %cycle) {
        $CYCLES{$item} = $cycle_ref;
    }
    #print join(":", sort keys %cycle) . "\n";
}

sub str_in_list ($@) {
    my ($str, @list) = @_;
    foreach my $item (@list) {
        return 1 if ($item eq $str);
    }
    return 0;
}
