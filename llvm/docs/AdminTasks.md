# LLVM Admin Tasks

## Grant Commit Access

- Frequency: Ongoing
- Permissions: Admin
- Description: An admin should periodically review the list of commit access requests.

This can be done by searching for issues in the llvm-project repository which have the [`infra:commit-access-request` label].
Any user that meets the [commit access requirements] should be added to the ['LLVM Committers' team].
The admin should add a comment to the issue when an invite to join the team has been sent and then close the issue when the invite has been accepted.
Admins should close issues after sending the invite and invite the requestor to reopen if there are issues.
If there are insufficient votes and it has been over two weeks since the last update, admins may remove the `infra:commit-access-request` label to take the request off the queue.
To reopen the request, someone with repository triage or write access must reapply the label to get it back on the dashboard.

[`infra:commit-access-request` label]: https://github.com/llvm/llvm-project/issues/?q=is%3Aissue%20state%3Aopen%20label%3Ainfra%3Acommit-access-request
[commit access requirements]: https://llvm.org/docs/DeveloperPolicy.html#obtaining-commit-access
['LLVM Committers' team]: https://github.com/orgs/llvm/teams/llvm-committers

## Commit Access Review

- Frequency: Monthly
- Permissions: Triage
- Description: The [Commit Access Review](https://github.com/llvm/llvm-project/actions/workflows/commit-access-review.yml)
  github actions job runs once per month and generates an artifact called 'triagers' which is a list of GitHub users who
  no longer qualify for commit access due to limited activity over the last 12 months.

When the job completes, an admin should download the triagers artifact and create an issue asking everyone on
the list if they still need commit access. Here is an [example issue](https://github.com/llvm/llvm-project/issues/131262).
The issue should have the 'infra:commit-access' label and should use the same description as the example issue.

## Removing Users from LLVM Committers team

- Frequencey: Monthly
- Permissions: Admin
- Description: Each month an admin should review any open issues with the infra:commit-access label.
  For any issue that has been open for more than 4 weeks, any user who has not responded on the ticket
  should be moved from the LLVM Committers team to the LLVM Triagers team.

## Action Secret Rotation

- Frequency: Monthly
- Permissions: Admin
- Description: Each month an admin should rotate the secrets that are used for GitHub Actions workflows
  in the llvm-project repository.

The secrets are personal access tokens that are associated with the llvmbot GitHub account. An admin
should log in to the llvmbot account, re-generate the personal access tokens and then copy the
new values into the corresponding secret in the llvm-project repo's settings.
